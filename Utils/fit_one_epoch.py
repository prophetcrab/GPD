import os

import torch
import torch.distributed as dist
from jinja2 import optimizer
from torch.utils.data import dataloader
from tqdm import tqdm

from Utils.utils import *
from Utils.LossAdmin import *

def fit_one_epoch(

        diffusion_model, diffusion_model_train,
        dataloader, optimizer,
        cuda,
        epoch_step, epoch, Epoch, save_period, save_dir,
        fp16, scaler,
        loss_output_path, loss_img_output_path,

):
    total_loss = 0
    """
    diffusion_model : 用来更新的diffusion_model
    diffusion_model_train : 用来训练的diffusion_model
    cuda:是否在cuda上训练
    optimizer:优化器
    epoch_step ：参数指定的进度条总步数，每个epoch需要执行的步骤数量，也就是每个epoch要进行训练的图像数量
    epoch：当前epoch
    Epoch：需要训练的总Epoch
    fp16：是否使用fp16精度
    scaler: 用于fp16精度训练
    save_period : 每隔save_period个世代就保存一下
    save_dir:保存的路径
    loss_output_path:loss数值储存的文件夹
    loss_img_output_path:loss折线图储存的文件夹
    :return: 
    """

    print('start train')
    pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    for iteration, data in enumerate(dataloader):
        if iteration >= epoch_step:
            break

        with torch.no_grad():

            if cuda:
                data = data.cuda()

        if not fp16:
            optimizer.zero_grad()
            diffusion_loss = torch.mean(diffusion_model_train(data))
            diffusion_loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            optimizer.zero_grad()
            with autocast():
                diffusion_loss = torch.mean(diffusion_model_train(data))
            scaler.scale(diffusion_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        diffusion_model.update_ema()

        total_loss += diffusion_loss.item()
        pbar.set_postfix(**{'totla_loss': total_loss / (iteration + 1),
                            'lr': get_lr(optimizer)})
        pbar.update(1)

    total_loss = total_loss / epoch_step
    pbar.close()
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.8f ' % (total_loss))

    lossAdmin = LossAdmin(loss_output_path, loss_img_output_path)
    lossAdmin.write_loss(epoch, total_loss)
    lossAdmin.plot_loss()

    if (epoch + 1) % save_period == 0 or (epoch + 1) == Epoch:
        torch.save(diffusion_model.state_dict(),
                   os.path.join(save_dir, "Diffusion_Epoch%d-GLoss%.6f.pth" % (epoch + 1, total_loss)))

    torch.save(diffusion_model.state_dict(), os.path.join(save_dir, "diffusion_model_last_epoch_weights.pth"))



