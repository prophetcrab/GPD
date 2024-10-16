import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from functools import partial
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def extract(a, t, x_shape):
    """
    从a中提取t位置的数据并且reshape成x_shape的形状返回
    :return:
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class EMA():
    """EMA 优化器"""
    def __init__(self, decay):
        self.decay = decay

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.decay + (1 - self.decay) * new

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old, new = ema_params.data, current_params.data
            ema_params.data = self.update_average(old, new)

def get_model(Model, device, **kwargs):
    """
    :param Model: 网络模型
    :param device: 设备
    :param kwargs: 网络参数
    :return:
    """
    model = Model(**kwargs).to(device)
    return model

class GaussianDiffusion(nn.Module):

    def __init__(self,
                 model, input_shape, input_channels, betas, device, num_class=None, loss_type="l2", ema_decay=0.9999,
                 ema_start=2000, ema_update_rate=1):
        """
        :param model: Network
        :param input_shape: img_size / data_size
        :param input_channels: channels
        :param betas:
        :param device:
        :param num_class:
        :param loss_type: l1 or l2
        :param ema_decay:
        :param ema_start:
        :param ema_update_rate:
        """
        super().__init__()

        self.model = model
        self.device = device
        self.ema_model = deepcopy(model)

        self.ema = EMA(decay=ema_decay)
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.step = 0

        self.input_shape = input_shape
        self.input_channels = input_channels
        self.num_class = num_class

        # l1或l2 loss
        if loss_type not in ['l1', 'l2']:
            raise ValueError('loss_type must be either l1 or l2')

        self.loss_type = loss_type
        self.num_timesteps = len(betas)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        # 将alpas转换成tensor
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        # betas             [0.0001, 0.00011992, 0.00013984 ... , 0.02]
        self.register_buffer("betas", to_torch(betas))
        # alphas            [0.9999, 0.99988008, 0.99986016 ... , 0.98]
        self.register_buffer("alphas", to_torch(alphas))
        # alphas_cumprod    [9.99900000e-01, 9.99780092e-01, 9.99640283e-01 ... , 4.03582977e-05]
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        # sqrt(alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        # sqrt(1 - alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        # sqrt(1 / alphas)
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.model)


    @torch.no_grad()
    def remove_noise(self, x, t, y, use_ema=True):
        """
        从xt中获取xt-1
        :param x: xt
        :param t: 当前时间步
        :param y: None
        :param use_ema: 是否使用ema
        :return: xt-1
        """

        if use_ema:
            return (
                    (x - extract(self.remove_noise_coeff, t, x.shape) * self.ema_model(x, t, y)) *
                    extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
        else:
            return (
                    (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, y)) *
                    extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )

    @torch.no_grad()
    def sample(self, batch_size, device, y=None, use_ema=True):
        """
        从ddpm里sample最终结果并返回
        :param batch_size: 这里好像只能设为1
        :param device: 采样使用cpu或者gpu
        :param y: None
        :param use_ema: 是否使用ema
        :return: 从ddpm中采样的结果  [B, N, C]
        """
        if y is not None and batch_size != len(y):
            raise ValueError('sample batch size different from length of given y')

        x = torch.randn(batch_size, self.input_shape, self.input_channels, device=device)

        for t in tqdm(range(self.num_timesteps - 1, -1, -1), desc="Sampling: ", total=self.num_timesteps):
            t_batch = torch.tensor([t], device=device).repeat(batch_size) #[B, 1]
            x = self.remove_noise(x, t_batch, y, use_ema=use_ema)


            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)


        return x.cpu().detach()


    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None, use_ema=True):
        """
        :param batch_size:
        :param device:
        :param y:
        :param use_ema:
        :return: (num_steps, (B, N, C))
        """
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.input_shape, self.input_channels, device=device)
        diffusion_sequence = [x.cpu().detach()]

        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

            diffusion_sequence.append(x.cpu().detach())

        return diffusion_sequence

    def perturb_x(self, x, t, noise):
        """从x0获取加噪的xt的噪声图"""
        return (
                extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

    def get_losses(self, x, t, y):
        """
        :param x: 随机噪声图
        :param t:  时间步
        :param y:  None
        :return:
        """
        noise = torch.randn_like(x)
        perturbed_x = self.perturb_x(x, t, noise)

        estimated_noise = self.model(perturbed_x, t)

        if self.loss_type == "l1":
            loss = F.l1_loss(estimated_noise, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(estimated_noise, noise)

        return loss

    def forward(self, x, y=None):
        B, N, C = x.shape
        device = x.device
        t = torch.randint(0, self.num_timesteps, (B,), device=device)
        return self.get_losses(x, t, y)


