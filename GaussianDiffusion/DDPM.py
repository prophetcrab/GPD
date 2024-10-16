from PIL import Image
from torch import nn
from GaussianDiffusion.Diffusion import *
from Utils.utils import *
from GaussianDiffusion.DiffusionUtils import *
from NDPModel.Model import *

class Diffusion(object):
    _defaults = {
        # -----------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        # -----------------------------------------------#
        "model_path": r"D:\PythonProject2\GaussianDiffusionFrame\Logs\Model\Airplane_model_5th\diffusion_model_last_epoch_weights.pth",
        # -----------------------------------------------#
        #   输入的通道数
        # -----------------------------------------------#
        "channels":3,
        # -----------------------------------------------#
        #   输入的形状和大小的设置
        #   图片：（width, height） 点云：（Size）
        # -----------------------------------------------#
        "input_shape": 2048,
        # -----------------------------------------------#
        #   betas相关参数
        # -----------------------------------------------#
        "schedule": "cosine",
        "num_timesteps": 1000,
        "schedule_low": 1e-4,
        "schedule_high": 0.02,
        # -----------------------------------------------#
        #   神经网络Model超参数
        # -----------------------------------------------#
        "n_layers": 12,
        "hidden_dim": 64,
        "num_heads": 8,
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True,
    }

    # ---------------------------------------------------#
    #   初始化Diffusion
    # ---------------------------------------------------#

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value
        self.generate()

        show_config(**self._defaults)

    def generate(self):
        # ----------------------------------------#
        #   创建Diffusion模型
        # ----------------------------------------#
        if self.schedule == "cosine":
            betas = generate_cosine_schedule(self.num_timesteps)
        else:
            betas = generate_linear_schedule(
                self.num_timesteps,
                self.schedule_low * 1000 / self.num_timesteps,
                self.schedule_high * 1000 / self.num_timesteps,
            )

        if self.cuda == True:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.net = GaussianDiffusion(
            AttentionModel(n_layers=self.n_layers, hidden_dim=self.hidden_dim, num_heads=self.num_heads),
            input_shape = self.input_shape,
            input_channels = self.channels,
            betas = betas,
            device = torch.device(self.device),
        )

        self.net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.net = self.net.eval()
        print('{} model loaded'.format(self.model_path))

        if self.cuda:
            self.net = self.net.cuda()

    def generate_sample_result(self, save_path, batch_size=1, data_size=2048, channels=3):
        with torch.no_grad():

            randn_in = torch.randn(batch_size, data_size, channels).cuda() if self.cuda else torch.randn(batch_size, data_size, channels)

            sample = self.net.sample(batch_size=1, device=self.device, use_ema=False)

            sample = sample[0].cpu().data.numpy()

            np.savetxt(save_path, sample, delimiter=",")

            return sample

    def generate_sample_result_sequence(self, batch_size=1, data_size=2048, channels=3):

        with(torch.no_grad()):
            sample_sequence = self.net.sample_diffusion_sequence(batch_size=1, device=self.device, use_ema=False)

            return sample_sequence



