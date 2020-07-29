import torch
import torch.nn as nn

from network import blocks


class Generator(nn.Module):
    def __init__(self, im_size, device=torch.device('cuda')):
        super(Generator, self).__init__()

        self.device = device
        self.unet = UNet(n_channels=6, n_classes=2, upsample_mode='bicubic')

    def forward(self, src_img, target_lmark):
        src = torch.cat([src_img, target_lmark], dim=-3)

        return self.unet(src)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, upsample_mode='nearest'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.down1 = Down(n_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        factor = 2
        self.down5 = Down(512, 1024 // factor)
        self.down6 = Down(512, 1024 // factor)

        self.resup = blocks.ResidualBlockUpNew(512, 512, upsample_mode=upsample_mode, norm=nn.InstanceNorm2d)
        self.up1 = Up(1024, 512, upsample_mode=upsample_mode, norm=nn.InstanceNorm2d)
        self.up2 = Up(1024, 256, upsample_mode=upsample_mode, norm=nn.InstanceNorm2d)
        self.up3 = Up(512, 128, upsample_mode=upsample_mode, norm=nn.InstanceNorm2d)
        self.up4 = Up(256, 64, upsample_mode=upsample_mode, norm=nn.InstanceNorm2d)
        self.up5 = Up(128, 3, upsample_mode=upsample_mode, norm=nn.InstanceNorm2d, activation=None)

    def forward(self, x):
        s1 = self.down1(x)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        s4 = self.down4(s3)
        s5 = self.down5(s4)
        s6 = self.down5(s5)

        for_skip = self.resup(s6)
        x = self.up1(for_skip, s5)  # 512 / 512
        x = self.up2(x, s4)  # 256 / 256
        x = self.up3(x, s3)  # 128 / 128
        x = self.up4(x, s2)  # 64 / 64

        out = self.up5(x, s1)
        out = torch.tanh(out)

        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, stride=2, kernel_size=3, padding=1)),
            blocks.ResidualBlock(out_channels, out_channels, norm=nn.InstanceNorm2d),
            blocks.ResidualBlock(out_channels, out_channels, norm=nn.InstanceNorm2d),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels,
                 upsample_mode='nearest', norm=nn.BatchNorm2d, activation=nn.ReLU):
        super().__init__()

        self.layers = nn.Sequential(
            blocks.ResidualBlockUpNew(
                in_channels, out_channels,
                upsample_mode=upsample_mode,
                activation=activation, norm=norm
            )
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.layers(x)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        return self.conv(x)
