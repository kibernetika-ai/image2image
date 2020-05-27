import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResBlockDown, SelfAttention, ResBlock, ResBlockD, ResBlockUp, Padding, adaIN
import math
import sys
import os


# components
class Embedder(tf.keras.Model):
    def __init__(self, in_height):
        super(Embedder, self).__init__()

        self.relu = layers.LeakyReLU()

        # in 6*224*224
        self.pad = Padding(in_height)  # out 6*256*256
        self.resDown1 = ResBlockDown(6, 64)  # out 64*128*128
        self.resDown2 = ResBlockDown(64, 128)  # out 128*64*64
        self.resDown3 = ResBlockDown(128, 256)  # out 256*32*32
        self.self_att = SelfAttention(256)  # out 256*32*32
        self.resDown4 = ResBlockDown(256, 512)  # out 515*16*16
        self.resDown5 = ResBlockDown(512, 512)  # out 512*8*8
        self.resDown6 = ResBlockDown(512, 512)  # out 512*4*4
        # self.sum_pooling = nn.AdaptiveMaxPool2d((1, 1))  # out 512*1*1
        self.sum_pooling = layers.GlobalMaxPool2D()

    def call(self, inputs, training=None, **kwargs):
        x = inputs[0]
        y = inputs[1]
        out = layers.concatenate((x, y), axis=-3)  # out 6*224*224
        out = self.pad(out)  # out 6*256*256
        out = self.resDown1(out)  # out 64*128*128
        out = self.resDown2(out)  # out 128*64*64
        out = self.resDown3(out)  # out 256*32*32

        out = self.self_att(out)  # out 256*32*32

        out = self.resDown4(out)  # out 512*16*16
        out = self.resDown5(out)  # out 512*8*8
        out = self.resDown6(out)  # out 512*4*4

        out = self.sum_pooling(out)  # out 512*1*1
        out = self.relu(out)  # out 512*1*1
        out = tf.reshape(out, [-1, 512, 1])  # out B*512*1
        return out


class Generator(tf.keras.Model):
    P_LEN = 2 * (512 * 2 * 5 + 512 + 256 + 256 + 128 + 128 + 64 + 64 + 32 + 32)
    slice_idx = [0,
                 512 * 4,  # res1
                 512 * 4,  # res2
                 512 * 4,  # res3
                 512 * 4,  # res4
                 512 * 4,  # res5
                 512 * 2 + 256 * 2,  # resUp1
                 256 * 2 + 128 * 2,  # resUp2
                 128 * 2 + 64 * 2,  # resUp3
                 64 * 2 + 32 * 2,  # resUp4
                 32 * 2]  # last adain
    for i in range(1, len(slice_idx)):
        slice_idx[i] = slice_idx[i - 1] + slice_idx[i]

    def __init__(self, in_height, finetuning=False, e_finetuning=None):
        super(Generator, self).__init__()

        self.sigmoid = tf.keras.activations.sigmoid
        self.relu = layers.LeakyReLU()

        # in 3*224*224 for voxceleb2
        self.pad = Padding(in_height)  # out 3*256*256

        # Down
        self.resDown1 = ResBlockDown(3, 64, conv_size=9, padding_size=4)  # out 64*128*128
        self.in1 = tfa.layers.InstanceNormalization()

        self.resDown2 = ResBlockDown(64, 128)  # out 128*64*64
        self.in2 = tfa.layers.InstanceNormalization()

        self.resDown3 = ResBlockDown(128, 256)  # out 256*32*32
        self.in3 = tfa.layers.InstanceNormalization()

        self.self_att_Down = SelfAttention(256)  # out 256*32*32

        self.resDown4 = ResBlockDown(256, 512)  # out 512*16*16
        self.in4 = tfa.layers.InstanceNormalization()

        # Res
        # in 512*16*16
        self.res1 = ResBlock(512)
        self.res2 = ResBlock(512)
        self.res3 = ResBlock(512)
        self.res4 = ResBlock(512)
        self.res5 = ResBlock(512)
        # out 512*16*16

        # Up
        # in 512*16*16
        self.resUp1 = ResBlockUp(512, 256)  # out 256*32*32
        self.resUp2 = ResBlockUp(256, 128)  # out 128*64*64

        self.self_att_Up = SelfAttention(128)  # out 128*64*64

        self.resUp3 = ResBlockUp(128, 64)  # out 64*128*128
        self.resUp4 = ResBlockUp(64, 32, out_size=(in_height, in_height), scale=None, conv_size=3,
                                 padding_size=1)  # out 3*224*224
        # self.conv2d = nn.Conv2d(32, 3, 3, padding=1)
        self.conv2d = layers.Conv2D(3, 3)

        # self.p = nn.Parameter(torch.rand(self.P_LEN, 512).normal_(0.0, 0.02))
        self.p = tf.Variable(initial_value=tf.random.normal([self.P_LEN, 512], 0.0, 0.02))

        self.finetuning = finetuning
        self.psi = tf.Variable(tf.random.normal([self.P_LEN, 1]))
        self.e_finetuning = e_finetuning

    def finetuning_init(self):
        if self.finetuning:
            self.psi = tf.Variable(tf.matmul(self.p, tf.reduce_mean(self.e_finetuning.mean, axis=0)))

    def call(self, inputs, training=None, **kwargs):
        y = inputs[0]
        e = inputs[1]
        if math.isnan(self.p[0, 0]):
            sys.exit()

        if self.finetuning:
            # e_psi = self.psi.unsqueeze(0)
            e_psi = tf.expand_dims(self.psi, axis=0)
            e_psi = e_psi.expand(e.shape[0], self.P_LEN, 1)
        else:
            # p = self.p.unsqueeze(0)
            p = tf.expand_dims(self.p, axis=0)
            # tf.broadcast_to()
            p = p.expand(e.shape[0], self.P_LEN, 512)
            e_psi = tf.matmul(p, e)  # B, p_len, 1

        # in 3*224*224 for voxceleb2
        out = self.pad(y)

        # Encoding
        out = self.resDown1(out)
        out = self.in1(out)

        out = self.resDown2(out)
        out = self.in2(out)

        out = self.resDown3(out)
        out = self.in3(out)

        out = self.self_att_Down(out)

        out = self.resDown4(out)
        out = self.in4(out)

        # Residual
        out = self.res1(out, e_psi[:, self.slice_idx[0]:self.slice_idx[1], :])
        out = self.res2(out, e_psi[:, self.slice_idx[1]:self.slice_idx[2], :])
        out = self.res3(out, e_psi[:, self.slice_idx[2]:self.slice_idx[3], :])
        out = self.res4(out, e_psi[:, self.slice_idx[3]:self.slice_idx[4], :])
        out = self.res5(out, e_psi[:, self.slice_idx[4]:self.slice_idx[5], :])

        # Decoding
        out = self.resUp1(out, e_psi[:, self.slice_idx[5]:self.slice_idx[6], :])

        out = self.resUp2(out, e_psi[:, self.slice_idx[6]:self.slice_idx[7], :])

        out = self.self_att_Up(out)

        out = self.resUp3(out, e_psi[:, self.slice_idx[7]:self.slice_idx[8], :])

        out = self.resUp4(out, e_psi[:, self.slice_idx[8]:self.slice_idx[9], :])

        out = adaIN(
            out,
            e_psi[:, self.slice_idx[9]:(self.slice_idx[10] + self.slice_idx[9]) // 2, :],
            e_psi[:, (self.slice_idx[10] + self.slice_idx[9]) // 2:self.slice_idx[10], :]
        )

        out = self.relu(out)

        out = self.conv2d(out)

        out = self.sigmoid(out)

        # out = out*255

        # out 3*224*224
        return out


# class W_i_class(nn.Module):
#     def __init__(self):
#         super(W_i_class, self).__init__()
#         self.W_i = nn.Parameter(torch.randn(512,2))

#     def forward(self):
#         return self.W_i

class Discriminator(tf.keras.Model):
    def __init__(self, finetuning=False, e_finetuning=None):
        super(Discriminator, self).__init__()
        self.relu = layers.LeakyReLU()

        # in 6*224*224
        self.pad = Padding(224)  # out 6*256*256
        self.resDown1 = ResBlockDown(6, 64)  # out 64*128*128
        self.resDown2 = ResBlockDown(64, 128)  # out 128*64*64
        self.resDown3 = ResBlockDown(128, 256)  # out 256*32*32
        self.self_att = SelfAttention(256)  # out 256*32*32
        self.resDown4 = ResBlockDown(256, 512)  # out 512*16*16
        self.resDown5 = ResBlockDown(512, 512)  # out 512*8*8
        self.resDown6 = ResBlockDown(512, 512)  # out 512*4*4
        self.res = ResBlockD(512)  # out 512*4*4
        # self.sum_pooling = nn.AdaptiveAvgPool2d((1, 1))  # out 512*1*1
        self.sum_pooling = layers.GlobalAvgPool2D()  # out 512*1*1

        # if not finetuning:
        #     print('Initializing Discriminator weights')
        #     if not os.path.isdir(self.path_to_Wi):
        #         os.mkdir(self.path_to_Wi)
        #     for i in tqdm(range(num_videos)):
        #         if not os.path.isfile(self.path_to_Wi + '/W_' + str(i) + '/W_' + str(i) + '.tar'):
        #             w_i = torch.rand(512, 1)
        #             os.mkdir(self.path_to_Wi + '/W_' + str(i))
        #             torch.save({'W_i': w_i}, self.path_to_Wi + '/W_' + str(i) + '/W_' + str(i) + '.tar')
        self.W_i = tf.Variable(tf.random.normal([512, 32]))
        self.w_0 = tf.Variable(tf.random.normal([512, 1]))
        self.b = tf.Variable(tf.random.uniform([]))

        self.finetuning = finetuning
        self.e_finetuning = e_finetuning
        self.w_prime = tf.Variable(tf.random.normal([512, 1]))

    def finetuning_init(self):
        if self.finetuning:
            self.w_prime = tf.Variable(self.w_0 + tf.reduce_mean(self.e_finetuning, axis=0))

    def load_W_i(self, W_i):
        self.W_i.data = self.relu(W_i)

    def call(self, inputs, training=None, **kwargs):
        x = inputs[0]
        y = inputs[1]
        i = inputs[2]
        # out = layers.concatenate([x, y], axis=-1)  # out B*6*224*224
        out = layers.concatenate([x, y], axis=-1)  # out B*224*224*6

        out = self.pad(out)

        out1 = self.resDown1(out)

        out2 = self.resDown2(out1)

        out3 = self.resDown3(out2)

        out = self.self_att(out3)

        out4 = self.resDown4(out)

        out5 = self.resDown5(out4)

        out6 = self.resDown6(out5)

        out7 = self.res(out6)

        out = self.sum_pooling(out7)

        out = self.relu(out)

        # out = out.squeeze(-1)  # out B*512*1
        out = tf.squeeze(out, axis=-1)

        # Calculate Realism Score
        _out = tf.transpose(out, [1, 2])
        _W_i = tf.transpose(tf.expand_dims(self.W[:, i], axis=-1), [0, 1])
        out = tf.matmul(_out, _W_i + self.w_0) + self.b
        out = tf.keras.activations.sigmoid(out)

        out = tf.reshape(out, x.shape[0])

        return out, [out1, out2, out3, out4, out5, out6, out7]


class Cropped_VGG19(nn.Module):
    def __init__(self):
        super(Cropped_VGG19, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, 3)
        self.conv1_2 = nn.Conv2d(64, 64, 3)
        self.conv2_1 = nn.Conv2d(64, 128, 3)
        self.conv2_2 = nn.Conv2d(128, 128, 3)
        self.conv3_1 = nn.Conv2d(128, 256, 3)
        self.conv3_2 = nn.Conv2d(256, 256, 3)
        self.conv3_3 = nn.Conv2d(256, 256, 3)
        self.conv4_1 = nn.Conv2d(256, 512, 3)
        self.conv4_2 = nn.Conv2d(512, 512, 3)
        self.conv4_3 = nn.Conv2d(512, 512, 3)
        self.conv5_1 = nn.Conv2d(512, 512, 3)
        # self.conv5_2 = nn.Conv2d(512,512,3)
        # self.conv5_3 = nn.Conv2d(512,512,3)

    def forward(self, x):
        conv1_1_pad = F.pad(x, (1, 1, 1, 1))
        conv1_1 = self.conv1_1(conv1_1_pad)
        relu1_1 = F.relu(conv1_1)
        conv1_2_pad = F.pad(relu1_1, (1, 1, 1, 1))
        conv1_2 = self.conv1_2(conv1_2_pad)
        relu1_2 = F.relu(conv1_2)
        pool1_pad = F.pad(relu1_2, (0, 1, 0, 1), value=float('-inf'))
        pool1 = F.max_pool2d(pool1_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv2_1_pad = F.pad(pool1, (1, 1, 1, 1))
        conv2_1 = self.conv2_1(conv2_1_pad)
        relu2_1 = F.relu(conv2_1)
        conv2_2_pad = F.pad(relu2_1, (1, 1, 1, 1))
        conv2_2 = self.conv2_2(conv2_2_pad)
        relu2_2 = F.relu(conv2_2)
        pool2_pad = F.pad(relu2_2, (0, 1, 0, 1), value=float('-inf'))
        pool2 = F.max_pool2d(pool2_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv3_1_pad = F.pad(pool2, (1, 1, 1, 1))
        conv3_1 = self.conv3_1(conv3_1_pad)
        relu3_1 = F.relu(conv3_1)
        conv3_2_pad = F.pad(relu3_1, (1, 1, 1, 1))
        conv3_2 = self.conv3_2(conv3_2_pad)
        relu3_2 = F.relu(conv3_2)
        conv3_3_pad = F.pad(relu3_2, (1, 1, 1, 1))
        conv3_3 = self.conv3_3(conv3_3_pad)
        relu3_3 = F.relu(conv3_3)
        pool3_pad = F.pad(relu3_3, (0, 1, 0, 1), value=float('-inf'))
        pool3 = F.max_pool2d(pool3_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv4_1_pad = F.pad(pool3, (1, 1, 1, 1))
        conv4_1 = self.conv4_1(conv4_1_pad)
        relu4_1 = F.relu(conv4_1)
        conv4_2_pad = F.pad(relu4_1, (1, 1, 1, 1))
        conv4_2 = self.conv4_2(conv4_2_pad)
        relu4_2 = F.relu(conv4_2)
        conv4_3_pad = F.pad(relu4_2, (1, 1, 1, 1))
        conv4_3 = self.conv4_3(conv4_3_pad)
        relu4_3 = F.relu(conv4_3)
        pool4_pad = F.pad(relu4_3, (0, 1, 0, 1), value=float('-inf'))
        pool4 = F.max_pool2d(pool4_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv5_1_pad = F.pad(pool4, (1, 1, 1, 1))
        conv5_1 = self.conv5_1(conv5_1_pad)
        relu5_1 = F.relu(conv5_1)

        return [relu1_1, relu2_1, relu3_1, relu4_1, relu5_1]
