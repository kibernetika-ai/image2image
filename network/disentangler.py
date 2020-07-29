import torch
from torch import nn
from torchvision import models


class Disentangler(nn.Module):
    def __init__(self):
        super(Disentangler, self).__init__()

        self.resnet50 = models.resnet50(pretrained=True, progress=False, num_classes=1000)
        self.fc1 = nn.Linear(2048 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 48)
        self.relu = nn.ReLU()

    def extract_features(self, img):
        # define hook
        def resnet_hook(module, input, output):
            output.detach_()  # no gradient compute
            resnet_features.append(output)

        resnet_features = []
        resnet_handlers = []

        # place hooks
        i = 4
        while hasattr(self.resnet50, f'layer{i}'):
            m = getattr(self.resnet50, f'layer{i}')[-1].bn3
            resnet_handlers.append(m.register_forward_hook(resnet_hook))
            i += 1

        # run model for x
        with torch.no_grad():
            self.resnet50(img)

        # retrieve features for x
        for h in resnet_handlers:
            h.remove()

        return resnet_features

    def forward(self, img, norm_landmark):
        features_raw = self.extract_features(img)

        num = features_raw[0].shape[1] * features_raw[0].shape[2] * features_raw[0].shape[3]
        features = features_raw[0].reshape([-1, num])
        for i in range(1, len(features_raw)):
            num = features_raw[i].shape[1] * features_raw[i].shape[2] * features_raw[i].shape[3]
            new_feature = features_raw[i].reshape([-1, num])
            features = torch.cat([features, new_feature], dim=1)

        # concat with norm landmark ??
        out = torch.cat([
            features, norm_landmark.reshape([-1, norm_landmark.shape[1] * norm_landmark.shape[2]])
        ], dim=1)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.relu(out)
        return out
