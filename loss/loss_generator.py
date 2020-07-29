import os
import sys

import importlib
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision.models import vgg19
from network import model


class CroppedVGG19(nn.Module):
    def __init__(self):
        super(CroppedVGG19, self).__init__()

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
        conv1_1_pad = F.pad(x, [1, 1, 1, 1])
        conv1_1 = self.conv1_1(conv1_1_pad)
        relu1_1 = F.relu(conv1_1)
        conv1_2_pad = F.pad(relu1_1, [1, 1, 1, 1])
        conv1_2 = self.conv1_2(conv1_2_pad)
        relu1_2 = F.relu(conv1_2)
        pool1_pad = F.pad(relu1_2, [0, 1, 0, 1], value=float('-inf'))
        pool1 = F.max_pool2d(pool1_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv2_1_pad = F.pad(pool1, [1, 1, 1, 1])
        conv2_1 = self.conv2_1(conv2_1_pad)
        relu2_1 = F.relu(conv2_1)
        conv2_2_pad = F.pad(relu2_1, [1, 1, 1, 1])
        conv2_2 = self.conv2_2(conv2_2_pad)
        relu2_2 = F.relu(conv2_2)
        pool2_pad = F.pad(relu2_2, [0, 1, 0, 1], value=float('-inf'))
        pool2 = F.max_pool2d(pool2_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv3_1_pad = F.pad(pool2, [1, 1, 1, 1])
        conv3_1 = self.conv3_1(conv3_1_pad)
        relu3_1 = F.relu(conv3_1)
        conv3_2_pad = F.pad(relu3_1, [1, 1, 1, 1])
        conv3_2 = self.conv3_2(conv3_2_pad)
        relu3_2 = F.relu(conv3_2)
        conv3_3_pad = F.pad(relu3_2, [1, 1, 1, 1])
        conv3_3 = self.conv3_3(conv3_3_pad)
        relu3_3 = F.relu(conv3_3)
        pool3_pad = F.pad(relu3_3, [0, 1, 0, 1], value=float('-inf'))
        pool3 = F.max_pool2d(pool3_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv4_1_pad = F.pad(pool3, [1, 1, 1, 1])
        conv4_1 = self.conv4_1(conv4_1_pad)
        relu4_1 = F.relu(conv4_1)
        conv4_2_pad = F.pad(relu4_1, [1, 1, 1, 1])
        conv4_2 = self.conv4_2(conv4_2_pad)
        relu4_2 = F.relu(conv4_2)
        conv4_3_pad = F.pad(relu4_2, [1, 1, 1, 1])
        conv4_3 = self.conv4_3(conv4_3_pad)
        relu4_3 = F.relu(conv4_3)
        pool4_pad = F.pad(relu4_3, [0, 1, 0, 1], value=float('-inf'))
        pool4 = F.max_pool2d(pool4_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv5_1_pad = F.pad(pool4, [1, 1, 1, 1])
        conv5_1 = self.conv5_1(conv5_1_pad)
        relu5_1 = F.relu(conv5_1)

        return [relu1_1, relu2_1, relu3_1, relu4_1, relu5_1]


class LossCnt(nn.Module):
    def __init__(self, vggface_body_path, vggface_weight_path, device):
        super(LossCnt, self).__init__()

        self.VGG19 = vgg19(pretrained=True)
        self.VGG19.eval()
        self.VGG19.to(device)

        # Inject vgg model code module as 'MainModel'
        dirname = os.path.dirname(vggface_body_path)
        sys.path.append(dirname)
        vgg_code = importlib.import_module(os.path.splitext(os.path.basename(vggface_body_path))[0])
        sys.modules['MainModel'] = vgg_code

        full_vgg_face = torch.load(vggface_weight_path, map_location='cpu')
        cropped_vgg_face = CroppedVGG19()
        cropped_vgg_face.load_state_dict(full_vgg_face.state_dict(), strict=False)
        self.VGGFace = cropped_vgg_face
        self.VGGFace.eval()
        self.VGGFace.to(device)

        self.l1_loss = nn.L1Loss()
        self.conv_idx_list = [3, 8, 13, 22, 31]  # idxes of relu layers in VGG19 cf.paper

    def forward(self, x, x_hat, vgg19_weight=1.5e-1, vggface_weight=2.5e-2):
        """Retrieve vggface feature maps"""
        with torch.no_grad():  # no need for gradient compute
            vgg_x_features = self.VGGFace(x)  # returns a list of feature maps at desired layers

        with torch.autograd.enable_grad():
            vgg_xhat_features = self.VGGFace(x_hat)

        lossface = 0
        for x_feat, xhat_feat in zip(vgg_x_features, vgg_xhat_features):
            lossface += self.l1_loss(x_feat, xhat_feat)

        """Retrieve vggface feature maps"""

        # define hook
        def vgg_x_hook(module, input, output):
            output.detach_()  # no gradient compute
            vgg_x_features.append(output)

        def vgg_xhat_hook(module, input, output):
            vgg_xhat_features.append(output)

        vgg_x_features = []
        vgg_xhat_features = []

        vgg_x_handles = []

        conv_idx_iter = 0

        # place hooks
        for i, m in enumerate(self.VGG19.features.modules()):
            if i == self.conv_idx_list[conv_idx_iter]:
                if conv_idx_iter < len(self.conv_idx_list) - 1:
                    conv_idx_iter += 1
                vgg_x_handles.append(m.register_forward_hook(vgg_x_hook))

        # run model for x
        with torch.no_grad():
            self.VGG19(x)

        # retrieve features for x
        for h in vgg_x_handles:
            h.remove()

        vgg_xhat_handles = []
        conv_idx_iter = 0

        # place hooks
        with torch.autograd.enable_grad():
            for i, m in enumerate(self.VGG19.features.modules()):
                if i == self.conv_idx_list[conv_idx_iter]:
                    if conv_idx_iter < len(self.conv_idx_list) - 1:
                        conv_idx_iter += 1
                    vgg_xhat_handles.append(m.register_forward_hook(vgg_xhat_hook))
            self.VGG19(x_hat)

            # retrieve features for x
            for h in vgg_xhat_handles:
                h.remove()

        loss19 = 0
        for x_feat, xhat_feat in zip(vgg_x_features, vgg_xhat_features):
            loss19 += self.l1_loss(x_feat, xhat_feat)

        loss = vgg19_weight * loss19 + vggface_weight * lossface

        return loss


class LossAdv(nn.Module):
    """
    Feature-matching loss
    """
    def __init__(self, fm_weight=10):
        super(LossAdv, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.FM_weight = fm_weight

    def forward(self, fake, d_real_list, d_fake_list):
        lossFM = 0
        for real, f in zip(d_real_list, d_fake_list):
            lossFM += self.l1_loss(real, f)

        return -fake.mean() + lossFM * self.FM_weight


class LossG(nn.Module):
    """
    Loss for generator meta training
    Inputs: x, x_hat
    output: lossG
    """

    def __init__(self, vggface_body_path, vggface_weight_path, device):
        super(LossG, self).__init__()

        self.perceptual = LossCnt(vggface_body_path, vggface_weight_path, device)
        self.l1_loss = nn.L1Loss()

    def forward(self, img, fake, ):
        perceptual = self.perceptual(img, fake)
        # print(perceptual.item(), loss_adv.item())
        # perceptual = Lp + Lpf
        # loss_adv = Lgan + Lfm
        return self.l1_loss(fake, img) + perceptual
