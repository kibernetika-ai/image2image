import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
import torch
import torch.nn as nn

from model_gan import spectral_norm


class ResBlockDown(layers.Layer):
    def __init__(self, in_channel, out_channel, conv_size=3, padding_size=1):
        super(ResBlockDown, self).__init__()

        self.relu = layers.LeakyReLU()
        self.relu_inplace = layers.LeakyReLU()
        self.avg_pool2d = layers.AvgPool2D(2)

        # left
        self.conv_l1 = spectral_norm.SpectralNormalization(layers.Conv2D(out_channel, 1))
        # self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, ))

        # right
        self.conv_r1 = spectral_norm.SpectralNormalization(layers.Conv2D(out_channel, conv_size))
        # self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding=padding_size))
        self.conv_r2 = spectral_norm.SpectralNormalization(layers.Conv2D(out_channel, conv_size))
        # self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding=padding_size))

    def call(self, x, **kwargs):
        res = x

        # left
        out_res = self.conv_l1(res)
        out_res = self.avg_pool2d(out_res)

        # right
        out = self.relu(x)
        out = self.conv_r1(out)
        out = self.relu_inplace(out)
        out = self.conv_r2(out)
        out = self.avg_pool2d(out)

        # merge
        out = out_res + out

        return out


class SelfAttention(layers.Layer):
    def __init__(self, in_channel):
        super(SelfAttention, self).__init__()

        # conv f
        # self.conv_f = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel // 8, 1))
        self.conv_f = spectral_norm.SpectralNormalization(layers.Conv2D(in_channel // 8, 1))
        # conv_g
        # self.conv_g = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel // 8, 1))
        self.conv_g = spectral_norm.SpectralNormalization(layers.Conv2D(in_channel // 8, 1))
        # conv_h
        # self.conv_h = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 1))
        self.conv_h = spectral_norm.SpectralNormalization(layers.Conv2D(in_channel, 1))

        self.softmax = layers.Softmax(axis=-2)  # sum in column j = 1
        self.gamma = tf.Variable(initial_value=1)

    def call(self, x, **kwargs):
        b, c, h, w = x.shape
        f_projection = self.conv_f(x)  # BxC'xHxW, C'=C//8
        g_projection = self.conv_g(x)  # BxC'xHxW
        h_projection = self.conv_h(x)  # BxCxHxW

        f_projection = tf.transpose(tf.reshape(f_projection, [b, -1, h * w]), [1, 2])  # BxNxC', N=H*W
        g_projection = tf.reshape(g_projection, [b, -1, h * w])  # BxC'xN
        h_projection = tf.reshape(h_projection, [b, -1, h * w])  # BxCxN

        attention_map = tf.matmul(f_projection, g_projection)  # BxNxN
        attention_map = self.softmax(attention_map)  # sum_i_N (A i,j) = 1

        # sum_i_N (A i,j) = 1 hence oj = (HxAj) is a weighted sum of input columns
        out = tf.matmul(h_projection, attention_map)  # BxCxN
        out = tf.reshape(out, [b, c, h, w])

        out = self.gamma * out + x
        return out


def adaIN(feature, mean_style, std_style, eps=1e-5):
    b, c, h, w = feature.shape

    feature = tf.reshape(feature, [b, c, -1])

    std_feat = tf.reshape(tf.math.reduce_std(feature, axis=2) + eps, [b, c, 1])
    mean_feat = tf.reshape(tf.reduce_mean(feature, axis=2), [b, c, 1])

    adain = std_style * (feature - mean_feat) / std_feat + mean_style

    adain = tf.reshape(adain, [b, c, h, w])
    return adain


class ResBlock(layers.Layer):
    def __init__(self, in_channel):
        super(ResBlock, self).__init__()

        # using no ReLU method

        # general
        self.relu = layers.LeakyReLU()

        # left
        # self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, padding=1))
        self.conv1 = spectral_norm.SpectralNormalization(layers.Conv2D(in_channel, 3))
        # self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, padding=1))
        self.conv2 = spectral_norm.SpectralNormalization(layers.Conv2D(in_channel, 3))

    def call(self, x, psi_slice):
        c = psi_slice.shape[1]

        res = x

        out = adaIN(x, psi_slice[:, 0:c // 4, :], psi_slice[:, c // 4:c // 2, :])
        out = self.relu(out)
        out = self.conv1(out)
        out = adaIN(out, psi_slice[:, c // 2:3 * c // 4, :], psi_slice[:, 3 * c // 4:c, :])
        out = self.relu(out)
        out = self.conv2(out)

        out = out + res

        return out


class ResBlockD(layers.Layer):
    def __init__(self, in_channel):
        super(ResBlockD, self).__init__()

        # using no ReLU method

        # general
        self.relu = layers.LeakyReLU()

        # left
        # self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, padding=1))
        self.conv1 = spectral_norm.SpectralNormalization(layers.Conv2D(in_channel, 3))
        # self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, padding=1))
        self.conv2 = spectral_norm.SpectralNormalization(layers.Conv2D(in_channel, 3))

    def forward(self, x):
        res = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out = out + res

        return out


class ResBlockUp(layers.Layer):
    def __init__(self, in_channel, out_channel, out_size=None, scale=2, conv_size=3, padding_size=1, is_bilinear=True):
        super(ResBlockUp, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        if is_bilinear:
            # self.upsample = nn.Upsample(size=out_size, scale_factor=scale, mode='bilinear')
            self.upsample = layers.UpSampling2D(size=scale, interpolation='bilinear')
        else:
            # self.upsample = nn.Upsample(size=out_size, scale_factor=scale)
            self.upsample = layers.UpSampling2D(size=scale)
        self.relu = layers.LeakyReLU()

        # left
        # self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1))
        self.conv_l1 = spectral_norm.SpectralNormalization(layers.Conv2D(out_channel, 1))

        # right
        # self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding=padding_size))
        self.conv_r1 = spectral_norm.SpectralNormalization(layers.Conv2D(out_channel, conv_size))
        # self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding=padding_size))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding=padding_size))

    def call(self, x, psi_slice):
        mean1 = psi_slice[:, 0:self.in_channel, :]
        std1 = psi_slice[:, self.in_channel:2 * self.in_channel, :]
        mean2 = psi_slice[:, 2 * self.in_channel:2 * self.in_channel + self.out_channel, :]
        std2 = psi_slice[:, 2 * self.in_channel + self.out_channel: 2 * (self.in_channel + self.out_channel), :]

        res = x

        # left
        out_res = self.upsample(res)
        out_res = self.conv_l1(out_res)

        # right
        out = adaIN(x, mean1, std1)
        out = self.relu(out)
        out = self.upsample(out)
        out = self.conv_r1(out)
        out = adaIN(out, mean2, std2)
        out = self.relu(out)
        out = self.conv_r2(out)

        out = out + out_res

        return out


class Padding(layers.Layer):
    def __init__(self, in_shape):
        super(Padding, self).__init__()

        # self.zero_pad = nn.ZeroPad2d(self.find_pad_size(in_shape))
        self.zero_pad = layers.ZeroPadding2D(self.find_pad_size(in_shape))

    def forward(self, x):
        out = self.zero_pad(x)
        return out

    def find_pad_size(self, in_shape):
        if in_shape < 256:
            pad_size = (256 - in_shape) // 2
        else:
            pad_size = 0
        return pad_size
