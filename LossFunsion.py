# from matplotlib import image
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
# from utils.utils_color import RGB_HSV, RGB_YCbCr
from loss_ssim import ssim
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import torch
import torch.fft
from math import exp

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)

def MSE(img1, img2, window_size=9):
    max_val = 255
    min_val = 0
    padd = window_size // 2
    (_, channel, height, width) = img1.size()
    img1_f = F.unfold(img1, (window_size, window_size), padding=padd)
    img2_f = F.unfold(img2, (window_size, window_size), padding=padd)
    res = (img1_f - img2_f) ** 2
    res = torch.sum(res, dim=1, keepdim=True) / (window_size ** 2)
    res = F.fold(res, output_size=(height, width), kernel_size=(1, 1))
    return res

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)                            # sigma = 1.5    shape: [11, 1]
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)    # unsqueeze()函数,增加维度  .t() 进行了转置 shape: [1, 1, 11, 11]
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()   # window shape: [1,1, 11, 11]
    return window

def std(img,  window_size=9):
    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma1 = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq
    return sigma1

class L_int(nn.Module):
    def __init__(self):
        super(L_int, self).__init__()
    def forward(self,img_vis, img_ir, img_fuse, mask=None):
        mse_ir = MSE(img_ir, img_fuse)
        mse_vi = MSE(img_vis, img_fuse)
        std_ir = std(img_ir)
        std_vi = std(img_vis)
        zero = torch.zeros_like(std_ir)
        one = torch.ones_like(std_vi)
        map1 = torch.where((std_ir - std_vi) > 0, one, zero)
        map_ir = torch.where(map1 + mask > 0, one, zero)
        map_vi = 1 - map_ir
        res = map_ir * mse_ir + map_vi * mse_vi
        return res.mean()

class fusion_loss(nn.Module):
    def __init__(self):
        super(fusion_loss, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Int = L_int()
    def forward(self, image_A, image_B, image_fused, mask):
        loss_l1 = 1 * self.L_Int(image_A, image_B, image_fused, mask)
        loss_gradient = 5 * self.L_Grad(image_A, image_B, image_fused)
        fusion_total = loss_l1 + loss_gradient
        return fusion_total, loss_gradient, loss_l1

if __name__ == '__main__':
    x1 = torch.tensor(np.random.rand(2, 1, 128, 128).astype(np.float32)).cuda()
    x = torch.tensor(np.random.rand(2, 1, 128, 128).astype(np.float32)).cuda()
    y = torch.tensor(np.random.rand(2, 1, 128, 128).astype(np.float32)).cuda()
    m = torch.tensor(np.random.rand(2, 1, 128, 128).astype(np.float32)).cuda()
    Fusion_loss = fusion_loss()
    L_T, L_G, L_I = Fusion_loss(x, x1, y, m)
    print(L_T)
    print(L_G)
    print(L_I)
