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
        self.conv_l1 = spectral_norm.SpectralNormalization(layers.Conv2D(out_channel, 1, padding='same'))
        # self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, ))

        # right
        self.conv_r1 = spectral_norm.SpectralNormalization(layers.Conv2D(out_channel, conv_size, padding='same'))
        # self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding=padding_size))
        self.conv_r2 = spectral_norm.SpectralNormalization(layers.Conv2D(out_channel, conv_size, padding='same'))
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
        self.conv_f = spectral_norm.SpectralNormalization(layers.Conv2D(in_channel // 8, 1, padding='same'))
        # conv_g
        # self.conv_g = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel // 8, 1))
        self.conv_g = spectral_norm.SpectralNormalization(layers.Conv2D(in_channel // 8, 1, padding='same'))
        # conv_h
        # self.conv_h = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 1))
        self.conv_h = spectral_norm.SpectralNormalization(layers.Conv2D(in_channel, 1, padding='same'))

        self.softmax = layers.Softmax()  # sum in column j = 1
        self.gamma = tf.Variable(initial_value=1.0)

    @staticmethod
    def hw_flatten(x):
        _, h, w, c = x.get_shape().as_list()
        return tf.reshape(x, shape=[-1, h * w, c])

    def call(self, x, **kwargs):
        b, h, w, c = x.shape
        f_projection = self.conv_f(x)  # BxHxWxC', C'=C//8
        g_projection = self.conv_g(x)  # BxHxWxC'
        h_projection = self.conv_h(x)  # BxHxWxC

        s = tf.matmul(self.hw_flatten(g_projection), self.hw_flatten(f_projection), transpose_b=True)

        # attention_map = tf.matmul(f_projection, g_projection)  # BxNxN
        attention_map = self.softmax(s)  # sum_i_N (A i,j) = 1
        # sum_i_N (A i,j) = 1 hence oj = (HxAj) is a weighted sum of input columns
        out = tf.matmul(attention_map, self.hw_flatten(h_projection))  # BxCxN
        out = tf.reshape(out, [-1, h, w, c])

        out = self.gamma * out + x
        return out


def adaIN(feature, mean_style, std_style, eps=1e-5):
    b, h, w, c = feature.shape

    feature = tf.reshape(feature, [b, -1, c])

    std_feat = tf.reshape(tf.math.reduce_std(feature, axis=1) + eps, [b, 1, c])
    mean_feat = tf.reshape(tf.reduce_mean(feature, axis=1), [b, 1, c])

    adain = std_style * (feature - mean_feat) / std_feat + mean_style

    print(adain.shape)
    adain = tf.reshape(adain, [b, h, w, c])
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

        self.upsample = layers.Conv2DTranspose(in_channel, 3, scale)
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
