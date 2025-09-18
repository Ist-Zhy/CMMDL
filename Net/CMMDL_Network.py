import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.fft as fft
from einops import rearrange
import os
import cv2
import torch.nn.init as init
import numbers

def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

class MDAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MDAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_conv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                  padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.q1X1_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.q1X1_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_conv = nn.Conv2d(dim*2, dim*2, kernel_size=3,
                                 padding=1, groups=dim*2,bias=bias)
        self.project_out_f = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_conv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))


        x_fft = fft.fftn(x, dim=(-2, -1)).real
        x_fft1 = self.q1X1_1(x_fft)
        x_fft2 = F.gelu(x_fft1)
        x_fft3 = self.q1X1_2(x_fft2)
        vf = fft.ifftn(x_fft3,dim=(-2, -1)).real

        kf, qf = self.kv_conv(self.kv(out)).chunk(2, dim=1)
        qf = qf.reshape(b, self.num_heads, -1, h * w)
        kf = kf.reshape(b, self.num_heads, -1, h * w)
        vf = vf.reshape(b, self.num_heads, -1, h * w)
        qf, kf = F.normalize(qf, dim=-1), F.normalize(kf, dim=-1)
        attnf = torch.softmax(torch.matmul(qf, kf.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out_f = self.project_out_f(torch.matmul(attnf, vf).reshape(b, -1, h, w))
        return out_f + out

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class MDAttention_T(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MDAttention_T, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_conv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                  padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.q1X1_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.q1X1_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_conv = nn.Conv2d(dim*2, dim*2, kernel_size=3,
                                 padding=1, groups=dim*2,bias=bias)
        self.project_out_f = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.patch_size = 8
        self.norm_IR = LayerNorm(dim, LayerNorm_type='WithBias')
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_conv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))

        x_fft = fft.fftn(x, dim=(-2, -1)).real
        x_fft1 = self.q1X1_1(x_fft)
        x_fft2 = F.gelu(x_fft1)
        x_fft3 = self.q1X1_2(x_fft2)
        qf = fft.ifftn(x_fft3,dim=(-2, -1)).real

        kf, vf = self.kv_conv(self.kv(out)).chunk(2, dim=1)
        # qf = qf.reshape(b, self.num_heads, -1, h * w)
        # kf = kf.reshape(b, self.num_heads, -1, h * w)
        # vf = vf.reshape(b, self.num_heads, -1, h * w)
        q_patch_1 = rearrange(vf, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch_1 = rearrange(kf, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                              patch2=self.patch_size)
        q_fft_1 = torch.fft.rfft2(q_patch_1.float())
        k_fft_1 = torch.fft.rfft2(k_patch_1.float())
        out_1 = q_fft_1 * k_fft_1
        out_1 = torch.fft.irfft2(out_1, s=(self.patch_size, self.patch_size))
        out_1 = rearrange(out_1, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                          patch2=self.patch_size)

        out_1 = self.norm_IR(out_1)
        out_1 = out_1 * qf
        # qf, kf = F.normalize(qf, dim=-1), F.normalize(kf, dim=-1)
        # attnf = torch.softmax(torch.matmul(qf, kf.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        # out_f = self.project_out_f(torch.matmul(attnf, vf).reshape(b, -1, h, w))
        return self.project_out_f(out_1) + out

class MDAttention_TT(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MDAttention_TT, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*4, kernel_size=1, bias=bias)
        self.qkv_conv = nn.Conv2d(dim * 4, dim * 4, kernel_size=3,
                                  padding=1, groups=dim * 4, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.q1X1_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.q1X1_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_conv = nn.Conv2d(dim*2, dim*2, kernel_size=3,
                                 padding=1, groups=dim*2,bias=bias)
        self.project_out_f = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.patch_size = 8
        self.norm_IR = LayerNorm(dim, LayerNorm_type='WithBias')
        self.norm_VI = LayerNorm(dim, LayerNorm_type='WithBias')
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_conv(self.qkv(x))
        q, k, v, x1 = qkv.chunk(4, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))

        q_fft = torch.fft.rfft2(x1.float())
        kf, vf = self.kv_conv(self.kv(out)).chunk(2, dim=1)
        k_fft = torch.fft.rfft2(kf.float())
        out_1 = q_fft * k_fft
        out_1 = torch.fft.irfft2(out_1, s=(h, w))
        out_1 = self.norm_IR(out_1)
        out_1 = out_1 * vf
        return self.project_out_f(out_1) + out



class MDAttention_TS(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MDAttention_TS, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_conv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                  padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.q1X1_1 = nn.Conv2d(2*dim, 2*dim, kernel_size=1, bias=False)
        self.q1X1_2 = nn.Conv2d(2*dim, 2*dim, kernel_size=1, bias=False)
        self.kv = nn.Conv2d(dim, dim*4, kernel_size=1, bias=bias)
        self.kv_conv = nn.Conv2d(dim*4, dim*4, kernel_size=3,
                                 padding=1, groups=dim*4,bias=bias)
        self.project_out_f = nn.Conv2d(2*dim, dim, kernel_size=1, bias=bias)
        self.xq = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.xq_conv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3,
                                  padding=1, groups=dim * 2, bias=bias)
        self.patch_size = 8
        self.norm_IR = LayerNorm(2*dim, LayerNorm_type='WithBias')
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_conv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))

        x = self.xq_conv(self.xq(x))
        x_fft = fft.fftn(x, dim=(-2, -1)).real
        x_fft1 = self.q1X1_1(x_fft)
        x_fft2 = F.gelu(x_fft1)
        x_fft3 = self.q1X1_2(x_fft2)
        qf = fft.ifftn(x_fft3,dim=(-2, -1)).real

        kf, vf = self.kv_conv(self.kv(out)).chunk(2, dim=1)
        # qf = qf.reshape(b, self.num_heads, -1, h * w)
        # kf = kf.reshape(b, self.num_heads, -1, h * w)
        # vf = vf.reshape(b, self.num_heads, -1, h * w)
        q_patch_1 = rearrange(vf, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch_1 = rearrange(kf, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                              patch2=self.patch_size)
        q_fft_1 = torch.fft.rfft2(q_patch_1.float())
        k_fft_1 = torch.fft.rfft2(k_patch_1.float())
        out_1 = q_fft_1 * k_fft_1
        out_1 = torch.fft.irfft2(out_1, s=(self.patch_size, self.patch_size))
        out_1 = rearrange(out_1, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                          patch2=self.patch_size)

        out_1 = self.norm_IR(out_1)
        out_1 = out_1 * qf
        # qf, kf = F.normalize(qf, dim=-1), F.normalize(kf, dim=-1)
        # attnf = torch.softmax(torch.matmul(qf, kf.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        # out_f = self.project_out_f(torch.matmul(attnf, vf).reshape(b, -1, h, w))
        return self.project_out_f(out_1) + out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1, bn=True, relu=True):
        padding = ((kernel_size - 1) * dilation + 1) // 2
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                              bias=False if bn else True)
        self.bn = bn
        if bn:
            self.bnop = nn.BatchNorm2d(out_planes)
        self.relu = relu
        if relu:
            self.reluop = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bnop(x)
        if self.relu:
            x = self.reluop(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = MDAttention_TT(dim, num_heads, False)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

class F_SDomain(nn.Module):
    def __init__(self, inp_channels=1, oup_channels=1, dim=64,
                 num_blocks=[4, 4], heads = [8, 8, 8],
                 ffn_expansion_factor=2, bias=False):
        super(F_SDomain, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.share_encoder = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0],
                            ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[0])])
    def forward(self, x):
        x_inp = self.patch_embed(x)
        x_oup = self.share_encoder(x_inp)
        return x_oup

class Sobel_xy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1):
        super(Sobel_xy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobel_x = self.convx(x)
        sobel_y = self.convy(x)
        x = torch.abs(sobel_x) + torch.abs(sobel_y)
        return x

class GradientConvolutionalBlock(nn.Module):
    def __init__(self, inp, oup, expand_ration):
        super(GradientConvolutionalBlock, self).__init__()
        hidden_dim = int(inp * expand_ration)

        self.Sobel = Sobel_xy(inp)
        self.Identity = nn.Conv2d(inp, oup, 1, 1, bias=False)
        self.Pad = nn.ReflectionPad2d(1)
        self.Conv_1 = nn.Conv2d(inp, hidden_dim, 1, 1, bias=False)
        self.Relu_1 = nn.ReLU6(inplace=True)
        self.Conv_2 = nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False)
        self.Relu_2 = nn.ReLU6(inplace=True)
        self.Conv_3 = nn.Conv2d(hidden_dim, oup, 1, bias=False)

        initialize_weights_xavier([self.Conv_1, self.Conv_2, self.Conv_3], 0.1)

    def forward(self, x):
        out = self.Pad(self.Relu_1(self.Conv_1(x)))
        out = self.Relu_2(self.Conv_2(out))
        out = self.Conv_3(out)
        # out = out + self.Identity(self.Sobel(x))
        return out + x

class InvBlock(nn.Module):
    def __init__(self, channel_num, channel_split_num, expand_ratio):
        super(InvBlock, self).__init__()

        self.s = None
        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num
        self.clamp = 0.8
        self.theta_phi = GradientConvolutionalBlock(inp=self.split_len1, oup=self.split_len1, expand_ration=expand_ratio)
        self.theta_rho = GradientConvolutionalBlock(inp=self.split_len1, oup=self.split_len1, expand_ration=expand_ratio)
        self.theta_eta = GradientConvolutionalBlock(inp=self.split_len1, oup=self.split_len1, expand_ration=expand_ratio)

    def forward(self, x):
        z1, z2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        y1 = z1 + self.theta_phi(z1)
        self.s = self.clamp * (torch.sigmoid(self.theta_rho(y1)) * 2 - 1)
        y2 = z2.mul(torch.exp(self.s)) + self.theta_eta(y1)
        out = torch.cat((y1, y2), 1)
        return out

class SpatialDomain(nn.Module):
    def __init__(self, channels=64, num_layers=3):
        super(SpatialDomain, self).__init__()
        self.GCBInv = nn.Sequential(*[InvBlock(channel_num=channels, channel_split_num=channels//2, expand_ratio=2)
                                      for i in range(num_layers)])
    def forward(self, x):
        return self.GCBInv(x)

class FrequencyDomain(nn.Module):
    def __init__(self, channels):
        super(FrequencyDomain, self).__init__()
        self.pre1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.pre2 = nn.Conv2d(channels, channels, 1, 1, 0)

        self.pha_fuse = nn.Sequential(nn.Conv2d(2*channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.amp_IR = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.amp_VIS = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                   nn.Conv2d(channels, channels, 1, 1, 0))
        self.post_IR = nn.Conv2d(channels, channels, 3, 1, 1)
        self.post_VIS = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, VIS, IR):
        _, _, H, W = VIS.shape

        VIS = torch.fft.rfft2(self.pre1(VIS)+1e-8, norm= 'backward')
        IR = torch.fft.rfft2(self.pre2(IR)+1e-8, norm= 'backward')

        VIS_amp = torch.abs(VIS)
        VIS_pha = torch.angle(VIS)
        IR_amp = torch.abs(IR)
        IR_pha = torch.angle(IR)

        pha_fuse = self.pha_fuse(torch.cat([VIS_pha, IR_pha], 1))
        amp_VIS = self.amp_VIS(VIS_amp)
        amp_IR = self.amp_IR(IR_amp)

        real_VIS = amp_VIS * torch.cos(pha_fuse) + 1e-8
        imag_VIS = amp_VIS * torch.sin(pha_fuse) + 1e-8
        real_IR = amp_IR * torch.cos(pha_fuse) + 1e-8
        imag_IR = amp_IR * torch.sin(pha_fuse) + 1e-8

        out_VIS = torch.complex(real_VIS, imag_VIS) + 1e-8
        out_IR = torch.complex(real_IR, imag_IR) + 1e-8

        out_VIS = torch.abs(torch.fft.irfft2(out_VIS, s=(H, W), norm='backward'))
        out_IR = torch.abs(torch.fft.irfft2(out_IR, s=(H, W), norm='backward'))

        return self.post_VIS(out_VIS), self.post_IR(out_IR)

class FrequencyDomain_T(nn.Module):
    def __init__(self, channels):
        super(FrequencyDomain_T, self).__init__()
        self.pre1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.pre2 = nn.Conv2d(channels, channels, 1, 1, 0)

        self.pha_fuse = nn.Sequential(nn.Conv2d(2*channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.amp_IR = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.amp_VIS = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                   nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_IR = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_VIS = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                   nn.Conv2d(channels, channels, 1, 1, 0))

        self.post_IR = nn.Conv2d(channels, channels, 3, 1, 1)
        self.post_VIS = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, VIS, IR):
        _, _, H, W = VIS.shape

        VIS = torch.fft.rfft2(self.pre1(VIS)+1e-8, norm= 'backward')
        IR = torch.fft.rfft2(self.pre2(IR)+1e-8, norm= 'backward')

        VIS_amp = torch.abs(VIS)
        VIS_pha = torch.angle(VIS)
        IR_amp = torch.abs(IR)
        IR_pha = torch.angle(IR)

        pha_fuse = self.pha_fuse(torch.cat([VIS_pha, IR_pha], 1))
        amp_VIS = self.amp_VIS(VIS_amp)
        amp_IR = self.amp_IR(IR_amp)
        pha_IR = self.pha_IR(IR_pha)
        pha_VIS = self.pha_VIS(VIS_pha)

        real_VIS = amp_VIS * torch.cos(pha_fuse) + 1e-8
        real_VIS_1 = amp_VIS * torch.cos(pha_VIS) + 1e-8
        imag_VIS = amp_VIS * torch.sin(pha_fuse) + 1e-8
        imag_VIS_1 = amp_VIS * torch.sin(pha_VIS) + 1e-8
        real_IR = amp_IR * torch.cos(pha_fuse) + 1e-8
        real_IR_1 = amp_IR * torch.cos(pha_IR) + 1e-8
        imag_IR = amp_IR * torch.sin(pha_fuse) + 1e-8
        imag_IR_1 = amp_IR * torch.sin(pha_IR) + 1e-8

        out_VIS_1 = torch.complex(real_VIS_1, imag_VIS_1) + 1e-8
        out_IR_1 = torch.complex(real_IR_1, imag_IR_1) + 1e-8
        out_VIS = torch.complex(real_VIS, imag_VIS) + 1e-8
        out_IR = torch.complex(real_IR, imag_IR) + 1e-8

        out_VIS = torch.abs(torch.fft.irfft2(out_VIS, s=(H, W), norm='backward'))
        out_IR = torch.abs(torch.fft.irfft2(out_IR, s=(H, W), norm='backward'))
        out_VIS_1 = torch.abs(torch.fft.irfft2(out_VIS_1, s=(H, W), norm='backward'))
        out_IR_1 = torch.abs(torch.fft.irfft2(out_IR_1, s=(H, W), norm='backward'))

        return self.post_VIS(out_VIS+out_VIS_1), self.post_IR(out_IR+out_IR_1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class DSRM(nn.Module):
    def __init__(self, in_channel=64, out_channel=64):
        super(DSRM, self).__init__()
        self.block1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            BasicConv2d(2 * in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            BasicConv2d(3 * in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.block4 = nn.Sequential(
            BasicConv2d(4 * in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(torch.cat([x, x1], dim=1))
        x3 = self.block3(torch.cat([x, x1, x2], dim=1))
        out = self.block4(torch.cat([x, x1, x2, x3], dim=1))
        return out

class SFD_Fusion(nn.Module):
    def __init__(self, channels):
        super(SFD_Fusion, self).__init__()
        inter_channels = channels // 2
        self.SA = SpatialAttention()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        self.CA_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.CA_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid_1 = nn.Sigmoid()
        self.sigmoid_2 = nn.Sigmoid()
        self.DSRM = DSRM(in_channel=channels, out_channel=channels)
        self.pre1 = ConvBNReLU(channels*2, channels, kernel_size=1)
        self.pre2 = ConvBNReLU(channels*2, channels, kernel_size=1)
        self.pre3 = ConvBNReLU(channels*2, channels, kernel_size=1)
    def forward(self, F_VIS, F_IR, S_VIS, S_IR):
        _, c, _, _ = F_VIS.shape
        F_1 = torch.cat([F_VIS, S_IR], dim=1)
        F_2 = torch.cat([S_VIS, F_IR], dim=1)
        F_1 = self.pre1(F_1)
        F_2 = self.pre2(F_2)
        F_3 = torch.cat([F_1, F_2], dim=1)
        map_w = self.SA(F_3)
        map_sa = map_w * F_3
        map_sa = map_sa + F_3
        x1, x2 = torch.split(map_sa, c, dim =1)
        x3 = self.DSRM(self.pre3(map_sa))
        x1 = x1 * self.sigmoid_1(self.CA_1(x1))
        x2 = x2 * self.sigmoid_2(self.CA_2(x2))
        return x1+x2+x3

class MultiDomainLearningBlock(nn.Module):
    def __init__(self, channels=64):
        super(MultiDomainLearningBlock, self).__init__()
        self.FDomain = FrequencyDomain_T(channels)
        self.SDomain_VIS = SpatialDomain(channels)
        self.SDomain_IR = SpatialDomain(channels)
        self.FSI = SFD_Fusion(channels)

    def forward(self, VIS_Feature, IR_Feature):
        Fre_VIS, Fre_IR = self.FDomain(VIS_Feature, IR_Feature)
        Spa_VIS = self.SDomain_VIS(VIS_Feature)
        Spa_IR = self.SDomain_IR(IR_Feature)

        F_Feature = self.FSI(Fre_VIS, Fre_IR, Spa_VIS, Spa_IR)
        return F_Feature

class F_Recon(nn.Module):
    def __init__(self, inp_channels=1, oup_channels=1, dim=64,
                     num_blocks=[4, 4], heads=[8, 8],
                     ffn_expansion_factor=2, bias=False):
        super(F_Recon, self).__init__()

        self.share_decoder = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1],
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias)
                                             for i in range(num_blocks[1])])
        self.output = nn.Sequential(
            nn.Conv2d(dim, dim//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(dim//2, oup_channels, kernel_size=3, stride=1, padding=1, bias=bias),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_oup = self.share_decoder(x)
        x_oup = self.output(x_oup)
        return self.sigmoid(x_oup)

class CMMDL(nn.Module):
    def __init__(self):
        super(CMMDL, self).__init__()
        self.share_Encoder = F_SDomain()
        self.MDLB = MultiDomainLearningBlock()
        self.share_Decoder = F_Recon()
    def forward(self, VIS, IR):
        VIS_Fre = self.share_Encoder(VIS)
        IR_Fre = self.share_Encoder(IR)
        F_Fre = self.MDLB(VIS_Fre, IR_Fre)
        F_Out = self.share_Decoder(F_Fre)
        return F_Out

def feature_save(tensor,name,i):
    # tensor = torchvision.utils.make_grid(tensor.transpose(1,0))
    tensor = torch.mean(tensor,dim=1)
    inp = tensor.detach().cpu().numpy().transpose(1,2,0)
    inp = inp.squeeze(2)
    inp = (inp - np.min(inp)) / (np.max(inp) - np.min(inp))
    if not os.path.exists(name):
        os.makedirs(name)
    inp = cv2.applyColorMap(np.uint8(inp * 255.0),cv2.COLORMAP_JET)
    cv2.imwrite(name + '/' + str(i) + '.png', inp)

if __name__ == '__main__':
    gpu_devices = [2, 3]
    device = torch.device("cuda:" + str(gpu_devices[0]) if torch.cuda.is_available() else "cpu")
    ir = torch.randn((2, 1, 256, 256))
    vis = torch.randn((2, 1, 256, 256))
    model = CMMDL()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))
    out = model(vis, ir)
    print(out.shape)

