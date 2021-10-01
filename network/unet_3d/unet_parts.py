""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, int(mid_channels)),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, int(out_channels)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        # diffY = x2.size()[3] - x1.size()[3]
        # diffX = x2.size()[4] - x1.size()[4]
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class Up2(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        # input is CHW
        # diffY = x2.size()[3] - x1.size()[3]
        # diffX = x2.size()[4] - x1.size()[4]
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x1, x2, x3], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ASPP(nn.Module):
    def __init__(self, input_channels, out_channels, scale_factor, dilations=[1, 3, 6, 9]):
        super(ASPP, self).__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.aspp0 = nn.Sequential(
            nn.Conv3d(input_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True))
        self.aspp1 = nn.Sequential(
            nn.Conv3d(input_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=dilations[0], bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv3d(input_channels, out_channels, kernel_size=3, stride=1, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv3d(input_channels, out_channels, kernel_size=3, stride=1, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv3d(input_channels, out_channels, kernel_size=3, stride=1, padding=dilations[3], dilation=dilations[3], bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.concat_process = nn.Sequential(
            nn.Conv3d(out_channels * 5, out_channels, 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

    def forward(self, x):
        d = x.shape[2]
        w, h = x.shape[3], x.shape[4]
        global_avg_pool = nn.AdaptiveAvgPool3d((d, 1, 1))
        upsample = nn.Upsample(scale_factor=(1, w, h), mode='trilinear', align_corners=True)
        x0 = self.aspp0(global_avg_pool(x))
        x0 = upsample(x0)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x6 = torch.cat((x0, x1, x2, x3, x4), dim=1)
        out = self.concat_process(x6)
        return out

# input_data = mx.nd.ones(shape=(4,4))
# kernel = mx.nd.ones(shape=(3,3))
# conv = mx.gluon.nn.Conv2DTranspose(channels=1, kernel_size=(3,3))
# # see appendix for definition of `apply_conv`
# output_data = apply_conv(input_data, kernel, conv)
# print(output_data)
#
# conv = mx.gluon.nn.Conv2DTranspose(channels=1, kernel_size=(3,3))
# output_data = apply_conv(input_data, kernel, conv)
# print(output_data)