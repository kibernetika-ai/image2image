import math

import torch
import torch.nn as nn


class WarpAlignBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=True, another_resup=False):
        super(WarpAlignBlock, self).__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, 2, kernel_size=1, padding=0, bias=False))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channel * 2, in_channel, kernel_size=1, padding=0, bias=False))
        if another_resup:
            self.res_up = ResidualBlockUpNew(in_channel, out_channel, norm=nn.InstanceNorm2d, is_bilinear=bilinear)
        else:
            self.res_up = ResidualBlockUp(in_channel, out_channel, norm=nn.InstanceNorm2d, is_bilinear=bilinear)

    def forward(self, s, u):
        f_u = self.conv1(u)  # flow map has 2 channels
        k = s.shape[0] // u.shape[0]
        s_mean = s.reshape([k, u.shape[0], s.shape[1], s.shape[2], s.shape[3]]).mean(dim=0)
        pose_adapt = nn.functional.grid_sample(s_mean, f_u.permute([0, 2, 3, 1]))

        out = torch.cat([u, pose_adapt], dim=-3)
        out = self.conv2(out)
        out = self.res_up(out)

        return out


class ImageAttention(nn.Module):
    def __init__(self, d_model=512, n_channels=64, image_size=256, device=None):
        super(ImageAttention, self).__init__()
        self.device = device if device else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.n_channels = n_channels
        self.wq = nn.Linear(d_model, n_channels)
        self.wk = nn.Linear(d_model, n_channels)
        self.wv = nn.Linear(d_model, d_model)
        self.wqp = nn.Linear(d_model, n_channels)
        self.wkp = nn.Linear(d_model, n_channels)
        self.norm1 = nn.InstanceNorm2d(d_model)
        self.norm2 = nn.InstanceNorm2d(d_model)
        self.conv3x3 = nn.utils.spectral_norm(conv3x3(d_model, d_model))

        # TODO positional encoding
        self.px = positionalencoding2d(d_model, image_size // 16, image_size // 16).permute([1, 2, 0]).unsqueeze(0).requires_grad_(False)
        self.py = positionalencoding2d(d_model, image_size // 32, image_size // 32).permute([1, 2, 0]).unsqueeze(0).requires_grad_(False)
        self.px = self.px.to(device)
        self.py = self.py.to(device)
        self.softmax = nn.Softmax(dim=-1)

    def attention(self, q, k, v):
        b_size = q.shape[0]
        channels = self.n_channels
        q_flatten = q.reshape([b_size, -1, channels])  # [B, 256, channels]
        k_flatten = k.reshape([b_size, -1, channels]).permute([0, 2, 1])  # [B, channels, 512]
        v_flatten = v.reshape([b_size, -1, v.shape[-1]])  # [B, 512, 512]

        out = torch.bmm(q_flatten, k_flatten) / math.sqrt(channels)  # [B, 256, 512]
        out = self.softmax(out).bmm(v_flatten)  # [B, 256, 512]
        out = out.reshape([b_size, q.shape[1], q.shape[2], v.shape[-1]])  # [B, 16, 16, 512]

        return out

    def forward(self, zx, zy):
        # zx: [B, 512, 16, 16]
        # zy: [BxK, 512, 8, 8]
        q = self.wq(zx.permute(0, 2, 3, 1)) + self.wqp(self.px)  # [B, 16, 16, 64]
        k = self.wk(zy.permute(0, 2, 3, 1)) + self.wkp(self.py)  # [BxK, 8, 8, 64]
        v = self.wv(zy.permute([0, 2, 3, 1]))  # [BxK, 8, 8, 512]

        out = self.attention(q, k, v).permute([0, 3, 1, 2])

        out += zx
        out_norm = self.norm1(out)
        out = self.conv3x3(out_norm)
        out += out_norm
        out = self.norm2(out)

        return out


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def positionalencoding2d(channels, height, width):
    """
    :param channels: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if channels % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(channels))
    pe = torch.zeros(channels, height, width)
    # Each dimension use half of d_model
    channels = int(channels // 4)

    # pe1 = torch.arange() * torch.sin(256 * i / (height * math.pow(10000, 2 * k / channels)))
    for i in range(height):
        for j in range(width):
            for k in range(0, channels):
                pe[4*k, i, j] = math.sin(256 * i / (height * math.pow(10000, 2 * k / channels)))
                pe[4*k+1, i, j] = math.cos(256 * i / (height * math.pow(10000, 2 * k / channels)))
                pe[4*k+2, i, j] = math.sin(256 * j / (width * math.pow(10000, 2 * k / channels)))
                pe[4*k+3, i, j] = math.cos(256 * j / (width * math.pow(10000, 2 * k / channels)))

    # div_term = torch.exp(torch.arange(0., channels, 2) *
    #                      -(math.log(10000.0) / channels))
    # pos_w = torch.arange(0., width).unsqueeze(1)
    # pos_h = torch.arange(0., height).unsqueeze(1)
    # pe[0:channels:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    # pe[1:channels:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    # pe[channels::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    # pe[channels + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    # __import__('ipdb').set_trace()

    return pe


class SinusoidalEncoding(nn.Module):
    def __init__(self):
        super(SinusoidalEncoding, self).__init__()

    def forward(self, x):
        return x


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.utils.spectral_norm(conv3x3(in_channels, out_channels, stride))
        if norm:
            if norm is nn.InstanceNorm2d:
                self.bn1 = norm(out_channels, affine=True)
                self.bn2 = norm(out_channels, affine=True)
            else:
                self.bn1 = norm(out_channels)
                self.bn2 = norm(out_channels)
        else:
            self.bn1 = None
            self.bn2 = None
        self.relu = nn.ReLU()
        self.conv2 = nn.utils.spectral_norm(conv3x3(out_channels, out_channels))
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.bn1:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.bn2:
            out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResidualDownBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, stride=2, norm=nn.BatchNorm2d):
        downsample = nn.Sequential(
            nn.utils.spectral_norm(conv3x3(in_channels, out_channels)),
            nn.AvgPool2d(2),
        )
        if norm is not None:
            if norm is nn.InstanceNorm2d:
                downsample.add_module('2', norm(out_channels, affine=True))
            else:
                downsample.add_module('2', norm(out_channels))
        super(ResidualDownBlock, self).__init__(
            in_channels, out_channels, stride=stride, downsample=downsample, norm=norm
        )


class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super(SelfAttention, self).__init__()

        # conv f
        self.conv_f = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel // 8, 1))
        # conv_g
        self.conv_g = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel // 8, 1))
        # conv_h
        self.conv_h = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 1))

        self.softmax = nn.Softmax(-2)  # sum in column j = 1
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        f_projection = self.conv_f(x)  # BxC'xHxW, C'=C//8
        g_projection = self.conv_g(x)  # BxC'xHxW
        h_projection = self.conv_h(x)  # BxCxHxW

        f_projection = torch.transpose(f_projection.view(B, -1, H * W), 1, 2)  # BxNxC', N=H*W
        g_projection = g_projection.view(B, -1, H * W)  # BxC'xN
        h_projection = h_projection.view(B, -1, H * W)  # BxCxN

        attention_map = torch.bmm(f_projection, g_projection)  # BxNxN
        attention_map = self.softmax(attention_map)  # sum_i_N (A i,j) = 1

        # sum_i_N (A i,j) = 1 hence oj = (HxAj) is a weighted sum of input columns
        out = torch.bmm(h_projection, attention_map)  # BxCxN
        out = out.view(B, C, H, W)

        out = self.gamma * out + x
        return out


# Residual block
class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, is_bilinear=True, norm=nn.BatchNorm2d):
        super(ResidualBlockUp, self).__init__()
        self.conv1 = nn.utils.spectral_norm(conv3x3(in_channels, out_channels))
        if norm is not None:
            if norm is nn.InstanceNorm2d:
                self.bn1 = norm(out_channels, affine=True)
                self.bn2 = norm(out_channels, affine=True)
                self.bn3 = norm(out_channels, affine=True)
            else:
                self.bn1 = norm(out_channels)
                self.bn2 = norm(out_channels)
                self.bn3 = norm(out_channels)
        else:
            self.bn1 = self.bn2 = self.bn3 = None

        self.relu = nn.ReLU()
        self.conv2 = nn.utils.spectral_norm(conv3x3(in_channels, out_channels))
        self.conv3 = nn.utils.spectral_norm(conv3x3(out_channels, out_channels))

        if is_bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.utils.spectral_norm(
                nn.ConvTranspose2d(in_channels, in_channels, kernel_size=1, stride=2, output_padding=1)
            )

    def forward(self, x):
        residual = x
        # left
        out_res = self.upsample(x)
        out_res = self.conv1(out_res)
        out_res = self.bn1(out_res)

        # right
        out = self.upsample(residual)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out = out + out_res
        out = self.relu(out)

        return out


# Residual block
class ResidualBlockUpNew(nn.Module):
    def __init__(self, in_channels, out_channels, is_bilinear=True, norm=nn.BatchNorm2d):
        super(ResidualBlockUpNew, self).__init__()
        self.conv1 = nn.utils.spectral_norm(conv3x3(in_channels, out_channels))
        if norm is not None:
            if norm is nn.InstanceNorm2d:
                self.bn1 = norm(out_channels, affine=True)
            else:
                self.bn1 = norm(out_channels)
        else:
            self.bn1 = None

        self.relu = nn.ReLU()
        self.res1 = ResidualBlock(out_channels, out_channels, norm=norm)
        self.res2 = ResidualBlock(out_channels, out_channels, norm=norm)

        if is_bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.upsample = nn.utils.spectral_norm(
                nn.ConvTranspose2d(in_channels, in_channels, kernel_size=1, stride=2, output_padding=1)
            )

    def forward(self, x):

        out_res = self.upsample(x)
        out_res = self.conv1(out_res)
        out_res = self.bn1(out_res)

        out = self.res1(out_res)
        out = self.res2(out)
        out = self.relu(out)

        return out


class Padding(nn.Module):
    def __init__(self, in_shape):
        super(Padding, self).__init__()

        self.zero_pad = nn.ZeroPad2d(self.findPadSize(in_shape))

    def forward(self, x):
        out = self.zero_pad(x)
        return out

    def findPadSize(self, in_shape):
        if in_shape < 256:
            pad_size = (256 - in_shape) // 2
        else:
            pad_size = 0
        return pad_size
