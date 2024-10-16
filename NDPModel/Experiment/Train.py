import datetime
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from NDPModel.Model import *
from NDPModel.NDPUtils import *

from Utils.utils import *
from Utils.fit_one_epoch import *

from DataUtils.DataProcessUtils import *
from DataUtils.DataSetUtils import *

from GaussianDiffusion.Diffusion import *
from GaussianDiffusion.DDPM import *



#-------------------------------------------------------------------------------------------#
#       注意！
#   1.在调整train的输入数据和模型数据后，需要去DDPM调整相关的数据来进行predict
#
#
#
#-------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    Cuda = True
    # ---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    # ---------------------------------------------------------------------#
    fp16 = False
    # ---------------------------------------------------------------------#
    #   如果想要断点续练就将model_path设置成logs文件夹下已经训练的权值文件。
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的。
    #   如果想要让模型从0开始训练，则设置model_path = ''。
    # ---------------------------------------------------------------------#
    diffusion_model_path = r"D:\PythonProject2\GaussianDiffusionFrame\Logs\Model\Airplane_model_5th\diffusion_model_last_epoch_weights.pth"
    # ---------------------------------------------------------------------#
    #   betas相关参数
    #   linear/cosine
    # ---------------------------------------------------------------------#
    schedule = "cosine"
    num_timesteps = 1000
    schedule_low = 1e-4
    schedule_high = 0.02
    # ---------------------------------------------------------------------#
    #   数据的大小 (N, 3)
    #   data_size为点云的长度 channels为点云通道数
    #   设置后在训练时Diffusion的图像看不出来，需要在预测时看单张图像。
    # ---------------------------------------------------------------------#
    data_size = 2048
    channels = 3
    # ------------------------------#
    #   训练参数设置
    # ------------------------------#
    Init_Epoch = 1625
    Epoch = 5000
    batch_size = 1
    # ------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    Init_lr = 2e-4
    Min_lr = Init_lr * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、adamw
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    # ------------------------------------------------------------------#
    optimizer_type = "adamw"
    momentum = 0.9
    weight_decay = 0
    # ------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    # ------------------------------------------------------------------#
    lr_decay_type = "cos"
    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    # ------------------------------------------------------------------#
    save_period = 50
    # ------------------------------------------------------------------#
    #   save_path        权值与日志文件保存的文件夹
    #   model_path       权值的储存文件夹
    #   loss_path        损失的储存文件夹
    #   train_label     训练的标签
    # ------------------------------------------------------------------#
    save_path = r'D:\PythonProject2\GaussianDiffusionFrame\Logs'
    loss_path = os.path.join(save_path, 'Loss')
    model_path = os.path.join(save_path, 'Model')
    train_label = "Airplane_model_5th"
    # ------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   内存较小的电脑可以设置为2或者0
    # ------------------------------------------------------------------#
    num_workers = 8
    # ------------------------------------------#
    #   获得点云数据的路径
    # ------------------------------------------#
    annotation_path = r"D:\PythonProject2\GaussianDiffusionFrame\dataList.txt"
    # ------------------------------------------#
    #   设置Attention网络参数
    #   n_layers : 多少层attention网络
    #   hidden_dim : attention的hidden_dim
    # ------------------------------------------#
    n_layers = 12
    hidden_dim = 64
    num_heads = 8
    # ------------------------------------------#
    #   设置device
    # ------------------------------------------#
    device = torch.device("cuda" if Cuda else "cpu")

    # ------------------------------------------------------#
    #   根据schedule方式获取betas
    # ------------------------------------------------------#
    if schedule == "cosine":
        betas = generate_cosine_schedule(num_timesteps)
    else:
        betas = generate_linear_schedule(
            num_timesteps,
            schedule_low * 1000 / num_timesteps,
            schedule_high * 1000 / num_timesteps,
        )
    # ------------------------------------------#
    #   Diffusion Model
    # ------------------------------------------#
    diffusion_model = GaussianDiffusion(
        AttentionModel(
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        ).to(device),
        input_shape=data_size,
        input_channels=channels,
        betas=betas,
        device=device,
    )

    # ------------------------------------------#
    #   将训练好的模型重新载入
    # ------------------------------------------#
    if diffusion_model_path != '':
        model_dict = diffusion_model.state_dict()
        pretrained_dict = torch.load(diffusion_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        diffusion_model.load_state_dict(model_dict)

    # ----------------------#
    #   记录Loss
    # ----------------------#
    loss_dir = os.path.join(loss_path, train_label+"_Loss")
    loss_output_dir = os.path.join(loss_dir, train_label+".txt")
    loss_img_output_dir = loss_dir
    save_dir = os.path.join(model_path, train_label)

    # ----------------------#
    #   判断目标文件夹是否存在
    # ----------------------#
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)
        print("loss dir created")
    else:
        print("loss dir already exists")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("save dir created")
    else:
        print("save dir already exists")
    # ------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    # ------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    diffusion_model_train = diffusion_model.train()

    if Cuda:
        cudnn.benchmark = True
        diffusion_model_train = torch.nn.DataParallel(diffusion_model)
        diffusion_model_train = diffusion_model_train.cuda()

    # ----------------------#
    #   训练数据数量
    # ----------------------#
    with open(annotation_path) as f:
        lines = f.readlines()
    num_train = len(lines)

    show_config(
        input_shape=data_size, Init_Epoch=Init_Epoch, Epoch=Epoch, batch_size=batch_size, \
        Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
        lr_decay_type=lr_decay_type, \
        save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train
    )

    # ------------------------------------------------------#
    #   Init_Epoch为起始世代
    #   Epoch总训练世代
    # ------------------------------------------------------#
    if True:
        # ---------------------------------------#
        #   根据optimizer_type选择优化器
        # ---------------------------------------#
        optimizer = {
            'adam': optim.Adam(diffusion_model_train.parameters(), lr=Init_lr, betas=(momentum, 0.999),
                               weight_decay=weight_decay),
            'adamw': optim.AdamW(diffusion_model_train.parameters(), lr=Init_lr, betas=(momentum, 0.999),
                                 weight_decay=weight_decay),
        }[optimizer_type]

        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, Epoch)

        # ---------------------------------------#
        #   判断每一个世代的长度
        # ---------------------------------------#
        epoch_step = num_train // batch_size
        if epoch_step == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        # ---------------------------------------#
        #   构建数据集加载器。
        # ---------------------------------------#
        train_dataset = PointDataSet(file_path_list=annotation_path,
                                     normalization=True,
                                     voxel_sample=True, voxel_size=0.001,
                                     random_sample=True, target_num_points=data_size)

        dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                pin_memory=True, drop_last=True, sampler=None)

        # ---------------------------------------#
        #   开始模型训练
        # ---------------------------------------#
        for epoch in range(Init_Epoch, Epoch):

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(

                diffusion_model, diffusion_model_train,
                dataloader, optimizer,
                Cuda,
                epoch_step, epoch, Epoch,save_period, save_dir,
                fp16, scaler,
                loss_output_dir, loss_img_output_dir

            )