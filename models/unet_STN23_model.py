import itertools

import torch
import torch.nn.functional as F

# from util.tb_visualizer import TensorboardVisualizer
from . import networks
from .base_model import BaseModel
import models.stn as stn
from Loss.pytorch_ssim import SSIM
from Loss import Depth_Aware_Loss
import numpy as np
import cv2
from .RetinalSeg import STN_Flow_relative

import random
import math


# def calculate_pos(grid,x,y,h,w):
#     dh = 2 / h
#     dw = 2 / w
#
#     new_x = grid[0,x,y,0]
#     new_y = grid[0,x,y,1]
#
#     # x = (x - (h-1)/2) * dh
#     # y = (y - (w-1)/2) * dw
#
#     ori_x = new_x/dh+(h-1)/2
#     ori_y = new_y/dw+(w-1)/2
#
#     return ori_y, ori_x

def GetGridPic(h=400, w=400, space=10):
    pic = -torch.ones((h, w)).cuda()
    for i in range(0, h):
        for j in range(0, w, space):
            pic[i][j] = 1
    for i in range(0, w):
        for j in range(0, h, space):
            pic[j][i] = 1
    pic = pic.unsqueeze(0).unsqueeze(0)
    return pic


def Gray(tensor):
    # TODO: make efficient
    b, c, h, w = tensor.shape
    R = tensor[0, 0, :, :]
    G = tensor[0, 1, :, :]
    B = tensor[0, 2, :, :]
    gray = torch.zeros((b, 1, h, w))
    gray[0] = 0.299 * R + 0.587 * G + 0.114 * B
    return gray


def Random_affine_grid(size=256):
    t = random.random() * 20 - 10
    t = torch.tensor([math.pi / 180 * t])
    scale = 1 + random.random() * 0.2 - 0.1
    dx = random.random() / 5 - 0.1
    dy = random.random() / 5 - 0.1
    theta = torch.tensor([[[scale * torch.cos(t), -torch.sin(t), dx],
                           [torch.sin(t), scale * torch.cos(t), dy],
                           [0, 0, 1]]])
    theta = theta.inverse()
    theta = theta[:, :2]
    grid = F.affine_grid(theta, size=(1, 1, size, size))
    return grid


class ChessBoardGenerator():
    def __init__(self, h=256, w=256, space=32):
        self.chessboard = torch.zeros((1, 1, h, w)).cuda()
        for i in range(0, h, 2 * space):
            for j in range(0, w, 2 * space):
                for k in range(0, space):
                    for l in range(0, space):
                        self.chessboard[:, :, i + k, j + l] = 1
        for i in range(space, h, 2 * space):
            for j in range(space, w, 2 * space):
                for k in range(0, space):
                    for l in range(0, space):
                        self.chessboard[:, :, i + k, j + l] = 1

    def __call__(self, img1, img2):
        return torch.where(self.chessboard == 1, img1, img2)


class UNETSTN23Model(BaseModel):
    """
    NeMAR: a neural multimodal adversarial image registration network.
    This class train a registration network and a geometry preserving translation network network. This is done
    using three networks:

    netT - A translation network that translates from modality A --to--> modality B (by default a
    netR - A registration network that applies geometric transformation to spatially align modality A --with--> modality B
    netD - Adversarial network that discriminates between fake an real images.

    Official implementation of:
    Unsupervised Multi-Modal Image Registration via Geometry Preserving Image-to-Image Translation paper
    https://arxiv.org/abs/2003.08073

    Inspired by the implementation of pix2pix:
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Modify the command line."""
        # parser = stn.modify_commandline_options(parser, is_train)
        if is_train:
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='Weight for the GAN loss.')
            parser.add_argument('--lambda_recon', type=float, default=100.0,
                                help='Weight for the L1 reconstruction loss.')
            parser.add_argument('--lambda_smooth', type=float, default=100.0,
                                help='Regularization term used by the STN')
            parser.add_argument('--lambda_depth', type=float, default=1.0,
                                help='Regularization term used by the STN')
            parser.add_argument('--depth_sum', type=bool, default=False,
                                help='Regularization term used by the STN')
            parser.add_argument('--enable_tbvis', action='store_true',
                                help='Enable tensorboard visualizer (default : False)')
            parser.add_argument('--multi_resolution', type=int, default=1,
                                help='Use of multi-resolution discriminator.'
                                     '(if equals to 1 then no multi-resolution training is applied)')
            # TensorboardVisualizer.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # Setup the visualizers
        self.train_stn = True
        self.setup_visualizers()
        # if self.isTrain and opt.enable_tbvis:
        #     self.tb_visualizer = TensorboardVisualizer(self, ['netR', 'netT', 'netD'], self.loss_names, self.opt)
        # else:
        #     self.tb_visualizer = None
        self.define_networks()
        # if self.tb_visualizer is not None:
        #     print('Enabling Tensorboard Visualizer!')
        #     self.tb_visualizer.enable()
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionSSIM = SSIM(window_size=11)
            self.setup_optimizers()

        self.GridPic = GetGridPic()
        self.chess_board_drawer = ChessBoardGenerator()
        self.test_index = 0

    def setup_visualizers(self):
        # <Loss>_TR denotes the loss for the translation first variant.
        # <Loss>_RT denotes the loss for the registration first variant.
        loss_names_A = ['L1_R', 'Depth_Aware', 'D', 'GAN']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # visual_names_A = ['registered_real_A', 'registered_seg_A','merge','merge_seg','grid','merge_A_segA','merge_reg_A_segA','seg_A','seg_B','merge_A_reg_A']
        # visual_names_A = ['registered_real_A', 'registered_seg_A', 'merge', 'merge_seg', 'grid', 'merge_reg_A_segA',
        #                   'seg_A', 'seg_B', 'merge_A_reg_A']
        visual_names_A = ['registered_affineReg_warp_real_B', 'grid1', 'warp_real_A', 'affineReg_warp_real_B',
                          'merge','merge_B']

        model_names_a = ['R']
        if self.isTrain:
            model_names_a += ['D']

        self.visual_names = ['real_A', 'real_B']
        self.model_names = []
        self.loss_names = []
        # if self.opt.direction == 'AtoB':
        self.visual_names += visual_names_A
        self.model_names += model_names_a
        self.loss_names += loss_names_A

    def define_networks(self):
        # define networks:
        # netT - is the photometric translation network (i.e the generator)
        # netR - is the registration network (i.e STN)
        # netD - is the discriminator network
        opt = self.opt
        # Support two directions (A->B) or (B->A)
        AtoB = opt.direction == 'AtoB'
        in_c = opt.input_nc if AtoB else opt.output_nc
        out_c = opt.output_nc if AtoB else opt.input_nc

        opt.input_nc = 1
        opt.output_nc = 1
        self.netR = stn.define_stn(self.opt, 'unet', padding='reflection')
        self.stn_destroy = STN_Flow_relative(size=(256, 256))
        self.netD = networks.define_D(2, opt.ndf, opt.netD,
                                      opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

    def reset_weights(self):
        # We have tested what happens if we reset the discriminator/translation network's weights during training.
        # This eventually will results in th
        pass

    def setup_optimizers(self):
        opt = self.opt

        # Define optimizer for the registration network:
        self.optimizer_R = torch.optim.Adam(itertools.chain(self.netR.parameters()),
                                            lr=opt.lr, betas=(opt.beta1, 0.999), )
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters()),
                                            lr=opt.lr, betas=(opt.beta1, 0.999), )

        self.optimizers.append(self.optimizer_R)
        self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        if AtoB:
            self.real_A = input['A'].to(self.device)
            self.real_B = input['B'].to(self.device)
            self.image_paths = input['A_paths']
        else:
            self.real_A = input['B'].to(self.device)
            self.real_B = input['A'].to(self.device)
            self.image_paths = input['B_paths']

        self.real_A = self.real_A.unsqueeze(0)
        self.real_B = self.real_B.unsqueeze(0)

        self.real_A = Gray(self.real_A).cuda()
        self.real_B = Gray(self.real_B).cuda()

        self.Grid_GT = Random_affine_grid().cuda()

        if (self.opt.random_registration):
            flow_r_src = np.array([cv2.resize(np.random.normal(scale=5, size=(3, 4)), dsize=(256, 256)),
                                   cv2.resize(np.random.normal(scale=5, size=(3, 4)), dsize=(256, 256))])
            flow_r_src = torch.cuda.FloatTensor(flow_r_src[np.newaxis], device=0)

            wrapped_imgs, _ = self.stn_destroy(flow_r_src, apply=[self.real_A, self.real_B], padding='reflection')
            self.warp_real_A = wrapped_imgs[0]
            self.warp_real_B_sameA = wrapped_imgs[1]

            flow_r_src = np.array([cv2.resize(np.random.normal(scale=5, size=(3, 4)), dsize=(256, 256)),
                                   cv2.resize(np.random.normal(scale=5, size=(3, 4)), dsize=(256, 256))])
            flow_r_src = torch.cuda.FloatTensor(flow_r_src[np.newaxis], device=0)

            wrapped_imgs, _ = self.stn_destroy(flow_r_src, apply=[self.real_B], padding='reflection')
            self.warp_real_B = wrapped_imgs[0]

            self.affineReg_warp_real_B = F.grid_sample(self.warp_real_B, self.Grid_GT, mode='bilinear',
                                                       padding_mode='reflection',
                                                       align_corners=False)
            self.affineReg_warp_real_A = F.grid_sample(self.warp_real_A, self.Grid_GT, mode='bilinear',
                                                       padding_mode='reflection',
                                                       align_corners=False)
        else:
            self.affineReg_warp_real_B = self.real_B
            self.affineReg_warp_real_A = self.real_A
            self.warp_real_A = self.real_A
            self.warp_real_B_sameA = self.real_B

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        # wraped_images, reg_term, self.field1 = self.netR(self.warp_real_A, self.affineReg_warp_real_B,
        #                                                  apply_on=[self.warp_real_A, self.warp_real_B_sameA, self.GridPic],
        #                                                  require_field=True)
        # self.registered_real_A = wraped_images[0]
        # self.registered_warp_real_B_sameA = wraped_images[1]
        # self.grid1 = wraped_images[2]

        wraped_images, reg_term, self.field1 = self.netR(self.warp_real_A, self.affineReg_warp_real_B,
                                                         apply_on=[self.affineReg_warp_real_B, self.GridPic],
                                                         require_field=True)
        self.registered_affineReg_warp_real_B = wraped_images[0]
        self.grid1 = wraped_images[1]


        if (self.opt.phase == 'test'):
            torch.save(self.field1, './results/' + self.opt.name + '/{}.pth'.format(self.test_index))
            self.test_index += 1

        self.merge = self.chess_board_drawer(self.warp_real_A, self.registered_affineReg_warp_real_B)
        self.merge_B = self.chess_board_drawer(self.warp_real_B_sameA, self.registered_affineReg_warp_real_B)

    def backward_T_and_R(self):
        """Calculate GAN and L1 loss for the translation and registration networks."""
        # Registration first (TR):
        # ----> Reconstruction loss:

        self.loss_L1_R = self.opt.lambda_recon * self.criterionL1(self.registered_affineReg_warp_real_B, self.warp_real_B_sameA)

        self.loss_Depth_Aware = self.opt.lambda_depth * Depth_Aware_Loss(self.field1, use_sum=self.opt.depth_sum)

        self.loss_GAN = self.opt.lambda_GAN * self.criterionGAN(self.netD(self.field1.permute(0, 3, 1, 2)), True)

        loss = self.loss_L1_R + self.loss_Depth_Aware + self.loss_GAN
        loss.backward()

        return loss

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        real = real.permute(0, 3, 1, 2)
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        fake = fake.detach().permute(0, 3, 1, 2)
        pred_fake = netD(fake)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        self.loss_D = self.backward_D_basic(self.netD, Random_affine_grid(), self.field1)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # TR(I_a) and RT(I_a)
        # Backward D
        self.set_requires_grad([self.netD], False)
        # Backward translation and registration networks
        self.optimizer_R.zero_grad()
        self.backward_T_and_R()  # calculate gradients for translation and registration networks
        self.optimizer_R.step()
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
