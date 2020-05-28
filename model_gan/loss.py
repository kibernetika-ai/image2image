import tensorflow as tf
from tensorflow.keras import layers
# import torch
# import torch.nn as nn
# import imp
# import torchvision
# from torchvision.models import vgg19
# from model_gan.model import Cropped_VGG19
from model_gan.vggface import VGGFace

vggface_feat_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
vggface_feat_layers2 = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
vgg19_feat_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']


def VGGFACE(input_tensor=None, input_shape=(224, 224, 3)):
    # vggface = VGGFace(input_tensor=input_tensor, model='vgg16', include_top=False, input_shape=input_shape)
    outputs = []
    vggface = tf.keras.applications.VGG16(input_tensor=input_tensor, include_top=False, input_shape=input_shape)
    for l in vggface_feat_layers2:
        outputs.append(vggface.get_layer(l).output)
    model = tf.keras.Model(inputs=vggface.input, outputs=outputs)

    return model


def VGG19(input_tensor=None, input_shape=(224, 224, 3)):
    vgg19 = tf.keras.applications.VGG19(input_tensor=input_tensor, include_top=False, input_shape=input_shape)
    outputs = []
    for l in vgg19_feat_layers:
        outputs.append(vgg19.get_layer(l).output)
    model = tf.keras.Model(inputs=vgg19.input, outputs=outputs)

    return model


def loss_dsc(r_x, r_x_hat):
    return tf.reduce_mean(tf.math.maximum(0.0, (1 + r_x_hat)) + tf.math.maximum(0.0, (1 - r_x)))


class LossCnt(object):
    def __init__(self, img_size):
        super(LossCnt, self).__init__()
        self.img_size = img_size
        self.loss_vgg19_wt = 1e-2
        self.loss_vggface_wt = 2e-3

        self.vggface = VGGFACE(
            input_tensor=None,
            input_shape=self.img_size
        )  # preprocess_tf_input(tf.multiply(tf.add(xi,1.0),127.5))

        self.vgg19 = VGG19(input_tensor=None, input_shape=self.img_size)

    def __call__(self, x, x_hat, vgg19_weight=1e-2, vggface_weight=2e-3):
        vggface_xi = self.vggface(x)
        vggface_x_hat = self.vggface(x_hat)
        vgg19_xi = self.vgg19(x)
        vgg19_x_hat = self.vgg19(x_hat)
        vggface_loss = 0
        for i in range(len(vggface_xi)):
            vggface_loss += tf.reduce_mean(tf.abs(vggface_xi[i] - vggface_x_hat[i]))

        vgg19_loss = 0
        for i in range(len(vgg19_xi)):
            vgg19_loss += tf.reduce_mean(tf.abs(vgg19_xi[i] - vgg19_x_hat[i]))

        return vgg19_loss * self.loss_vgg19_wt + vggface_loss * self.loss_vggface_wt


class LossAdv(object):
    def __init__(self, FM_weight=1e1):
        super(LossAdv, self).__init__()
        self.FM_weight = FM_weight

    def loss_fm(self, d_act, d_act_hat):
        loss = 0
        for i in range(0, len(d_act)):
            loss += tf.reduce_mean(tf.abs(d_act[i] - d_act_hat[i]))
        return loss * self.FM_weight

    def __call__(self, r_hat, d_act, d_act_hat):
        return -tf.reduce_mean(r_hat) + self.loss_fm(d_act, d_act_hat)


class LossMatch(object):
    def __init__(self, match_weight=8e1):
        super(LossMatch, self).__init__()
        # self.l1_loss = nn.L1Loss()
        self.match_weight = match_weight

    def __call__(self, e_vectors, W, i):
        return tf.reduce_mean(tf.abs(W - e_vectors)) * self.match_weight


class LossG(object):
    """
    Loss for generator meta training
    Inputs: x, x_hat, r_hat, D_res_list, D_hat_res_list, e, W, i
    output: lossG
    """

    def __init__(self, img_size):
        super(LossG, self).__init__()

        self.LossCnt = LossCnt(img_size)
        self.lossAdv = LossAdv()
        self.lossMatch = LossMatch()

    def __call__(self, x, x_hat, r_hat, D_res_list, D_hat_res_list, e_vectors, W, i):
        loss_cnt = self.LossCnt(x, x_hat)
        loss_adv = self.lossAdv(r_hat, D_res_list, D_hat_res_list)
        loss_match = self.lossMatch(e_vectors, W, i)
        return loss_cnt + loss_adv + loss_match


def loss_dscreal(r):
    loss = tf.math.maximum(tf.zeros_like(r), 1 - r)
    return tf.reduce_mean(loss)


def loss_dscfake(rhat):
    loss = tf.math.maximum(tf.zeros_like(rhat), 1 + rhat)
    return tf.reduce_mean(loss)


class LossGF(layers.Layer):
    """
    Loss for generator finetuning
    Inputs: x, x_hat, r_hat, D_res_list, D_hat_res_list, e, W, i
    output: lossG
    """

    def __init__(self, img_size):
        super(LossGF, self).__init__()

        self.LossCnt = LossCnt(img_size)
        self.lossAdv = LossAdv()

    def call(self, x, x_hat, r_hat, D_res_list, D_hat_res_list):
        loss_cnt = self.LossCnt(x, x_hat)
        loss_adv = self.lossAdv(r_hat, D_res_list, D_hat_res_list)
        return loss_cnt + loss_adv
