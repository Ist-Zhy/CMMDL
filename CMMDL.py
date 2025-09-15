import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.fft as fft
from einops import rearrange
import os
import cv2
import torch.nn.init as init

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
        qf = fft.ifftn(x_fft3,dim=(-2, -1)).real

        kf, vf = self.kv_conv(self.kv(out)).chunk(2, dim=1)
        qf = qf.reshape(b, self.num_heads, -1, h * w)
        kf = kf.reshape(b, self.num_heads, -1, h * w)
        vf = vf.reshape(b, self.num_heads, -1, h * w)
        qf, kf = F.normalize(qf, dim=-1), F.normalize(kf, dim=-1)
        attnf = torch.softmax(torch.matmul(qf, kf.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out_f = self.project_out_f(torch.matmul(attnf, vf).reshape(b, -1, h, w))

        return out_f

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

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = MDAttention(dim, num_heads, bias=False)
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
        self.Pad = nn.ReflectionPad2d(2)
        self.Conv_1 = nn.Conv2d(inp, hidden_dim, 3, 1, bias=False)
        self.Relu_1 = nn.ReLU6(inplace=True)
        self.Conv_2 = nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False)
        self.Relu_2 = nn.ReLU6(inplace=True)
        self.Conv_3 = nn.Conv2d(hidden_dim, oup, 1, bias=False)

        initialize_weights_xavier([self.Conv_1, self.Conv_2, self.Conv_3], 0.1)

    def forward(self, x):
        out = self.Pad(self.Relu_1(self.Conv_1(x)))
        out = self.Relu_2(self.Conv_2(out))
        out = self.Conv_3(out)
        out = out + self.Identity(self.Sobel(x))
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
        # self.GCBInv = InvBlock(channel_num=channels, channel_split_num=channels//2, expand_ratio=2)
        self.GCBInv = nn.Sequential(*[InvBlock(channel_num=channels, channel_split_num=channels//2, expand_ratio=2)
                                      for i in range(num_layers)])

    def forward(self, x):
        return self.GCBInv(x)

class FDomain(nn.Module):
    def __init__(self, channels):
        super(FDomain, self).__init__()
        self.pre1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.pre2 = nn.Conv2d(channels, channels, 1, 1, 0)

        self.amp_fuse = nn.Sequential(nn.Conv2d(2*channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(2*channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, VIS, IR):
        _, _, H, W = VIS.shape

        VIS = torch.fft.rfft2(self.pre1(VIS)+1e-8, norm= 'backward')
        IR = torch.fft.rfft2(self.pre2(IR)+1e-8, norm= 'backward')

        VIS_amp = torch.abs(VIS)
        VIS_pha = torch.angle(VIS)
        IR_amp = torch.abs(IR)
        IR_pha = torch.angle(IR)

        amp_fuse = self.amp_fuse(torch.cat([VIS_amp, IR_amp], 1))
        pha_fuse = self.pha_fuse(torch.cat([VIS_pha, IR_pha], 1))

        real = amp_fuse * torch.cos(pha_fuse) + 1e-8
        imag = amp_fuse * torch.sin(pha_fuse) + 1e-8

        out = torch.complex(real, imag) + 1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))

        return self.post(out)

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

class FSInteraction(nn.Module):
    def __init__(self, channels):
        super(FSInteraction, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels

        self.spa_att_VIS = nn.Sequential(nn.Conv2d(channels, channels // 2, 3, 1, 1, bias=True),
                                         nn.LeakyReLU(0.1),
                                         nn.Conv2d(channels // 2, channels, 3, 1, 1, bias=True),
                                         nn.Sigmoid())
        self.cha_att_VIS = nn.Sequential(nn.Conv2d(channels * 2, channels // 2, 1),
                                         nn.LeakyReLU(0.1),
                                         nn.Conv2d(channels // 2, channels * 2, 1),
                                         nn.Sigmoid())
        self.post_VIS = nn.Conv2d(channels * 2, channels, 3, 1, 1)

        self.spa_att_IR = nn.Sequential(nn.Conv2d(channels, channels // 2, 3, 1, 1, bias=True),
                                         nn.LeakyReLU(0.1),
                                         nn.Conv2d(channels // 2, channels, 3, 1, 1, bias=True),
                                         nn.Sigmoid())
        self.cha_att_IR = nn.Sequential(nn.Conv2d(channels * 2, channels // 2, 1),
                                         nn.LeakyReLU(0.1),
                                         nn.Conv2d(channels // 2, channels * 2, 1),
                                         nn.Sigmoid())
        self.post_IR = nn.Conv2d(channels * 2, channels, 3, 1, 1)
        self.post = nn.Conv2d(channels * 2, channels, 3, 1, 1)

    def forward(self, Fre_VIS, Fre_IR, Spa_F):
        VIS_map = self.spa_att_VIS(Spa_F - Fre_VIS)
        VIS_res = VIS_map * Spa_F + Spa_F
        VIS_cat = torch.cat([VIS_res, Fre_VIS], 1)
        VIS_cha = self.post_VIS(self.cha_att_VIS(self.contrast(VIS_cat) + self.avgpool(VIS_cat))*VIS_cat)
        VIS_out = VIS_cha + Spa_F

        IR_map = self.spa_att_IR(Spa_F - Fre_IR)
        IR_res = IR_map * Spa_F + Spa_F
        IR_cat = torch.cat([IR_res, Fre_IR], 1)
        IR_cha = self.post_IR(self.cha_att_IR(self.contrast(IR_cat) + self.avgpool(IR_cat)) * IR_cat)
        IR_out = IR_cha + Spa_F
        F_out = torch.cat([VIS_out, IR_out], 1)

        return self.post(F_out)

class FSI(nn.Module):
    def __init__(self, channels):
        super(FSI, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels

        self.spa_att = nn.Sequential(nn.Conv2d(channels, channels // 2, 3, 1, 1, bias=True),
                                         nn.LeakyReLU(0.1),
                                         nn.Conv2d(channels // 2, channels, 3, 1, 1, bias=True),
                                         nn.Sigmoid())
        self.cha_att = nn.Sequential(nn.Conv2d(channels * 2, channels // 2, 1),
                                         nn.LeakyReLU(0.1),
                                         nn.Conv2d(channels // 2, channels * 2, 1),
                                         nn.Sigmoid())
        self.post = nn.Conv2d(channels * 2, channels, 3, 1, 1)
        self.pre1 = nn.Conv2d(channels * 2, channels, 1)
        self.pre2 = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, F_VIS, F_IR, S_VIS, S_IR):
        F_1 = torch.cat([F_VIS, S_IR], dim=1)
        F_2 = torch.cat([S_VIS, F_IR], dim=1)
        # feature_save(F_1, 'V', 17)
        # feature_save(F_2, 'I', 18)
        F_1 = self.pre1(F_1)
        F_2 = self.pre2(F_2)
        # feature_save(F_1, 'V', 15)
        # feature_save(F_2, 'I', 16)
        # feature_save(F_2-F_1, 'V', 20)
        F_map = self.spa_att(F_2 - F_1)
        F_res = F_1 * F_map + F_2
        cat_f = torch.cat([F_res, F_1], dim=1)
        cha_res = self.post(self.cha_att(self.contrast(cat_f) + self.avgpool(cat_f)) * cat_f)
        # feature_save(cha_res, 'V', 21)
        return cha_res

class MultiDomainLearningBlock(nn.Module):
    def __init__(self, channels=64):
        super(MultiDomainLearningBlock, self).__init__()
        self.FDomain = FrequencyDomain(channels)
        self.SDomain_VIS = SpatialDomain(channels)
        self.SDomain_IR = SpatialDomain(channels)
        # self.FSI = FSInteraction(channels)
        self.FSI = FSI(channels)
        self.reduce = nn.Conv2d(2*channels, channels, 1, 1, 0)

    def forward(self, VIS_Feature, IR_Feature):
        Fre_VIS, Fre_IR = self.FDomain(VIS_Feature, IR_Feature)
        # feature_save(Fre_VIS, "V", 13)
        # feature_save(Fre_IR, 'I', 14)
        Spa_VIS = self.SDomain_VIS(VIS_Feature)
        Spa_IR = self.SDomain_IR(IR_Feature)
        # feature_save(Spa_VIS, "V", 11)
        # feature_save(Spa_IR, 'I', 12)
        # Spa_F = self.reduce(torch.cat([self.SDomain_VIS(VIS_Feature), self.SDomain_IR(IR_Feature)], 1))
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
    ir = torch.randn((2, 1, 128, 128))
    vis = torch.randn((2, 1, 128, 128))
    # model_ShareEncoder = F_SDomain()
    # ir_f = model_ShareEncoder(ir)
    # vis_f = model_ShareEncoder(vis)
    # model_MDLB = MultiDomainLearningBlock(64)
    # out = model_MDLB(vis_f, ir_f)
    # De = F_Recon()
    # out = De(out)
    model = CMMDL()
   # print(model)
    out = model(vis, ir)
    # print(torch.cuda.is_available())
    print(out.shape)

