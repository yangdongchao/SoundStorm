""" Parts of the U-Net model """
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, padding=1,
                bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=(1, 2)):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size),
            DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=(1, 2),
                 bilinear=True):
        super().__init__()
        self.scale_factor = scale_factor
        # if bilinear, use the normal convolutions to reduce the number of channels
        #if bilinear:
        #self.up = torch.nn.functional.interpolate(scale_factor=(1,2), mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        # else:
        #     self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        #     self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        print('x1 before inter ', x1.shape)
        x1 = torch.nn.functional.interpolate(
            x1, scale_factor=self.scale_factor, mode="nearest")
        print('x1 inter ', x1.shape)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        print('x1 ', x1.shape)
        print('x2 ', x2.shape)
        x = torch.cat([x2, x1], dim=1)
        print('up x ', x.shape)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.inc = (DoubleConv(n_channels, n_channels))
        self.down1 = (Down(n_channels, n_channels, kernel_size=(2, 1)))
        self.down2 = (Down(n_channels, n_channels, kernel_size=(2, 1)))
        self.down3 = (Down(n_channels, n_channels, kernel_size=(3, 1)))

        self.up1 = (Up(n_channels * 2, n_channels, scale_factor=(3, 1)))
        self.up2 = (Up(n_channels * 2, n_channels, scale_factor=(2, 1)))
        self.up3 = (Up(n_channels * 2, n_channels, scale_factor=(2, 1)))
        #self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)  # 256
        # print('x1 ', x1.shape)
        x2 = self.down1(x1)  # 256
        # print('x2 ', x2.shape)
        x3 = self.down2(x2)  # 256
        # print('x3 ', x3.shape)
        x4 = self.down3(x3)  # 256
        # print('x4 ', x4.shape)
        # x5 = self.down4(x4)
        x = self.up1(x4, x3)  # 512
        # print('up1 ', x.shape) # 256
        x = self.up2(x, x2)  # 256
        # print('up2 ', x.shape)
        x = self.up3(x, x1)
        # print('up3 ', x.shape)
        return x

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


unet = UNet(256)
x = torch.rand((2, 256, 12, 999))
x_out = unet(x)
print('x_out ', x_out.shape)
