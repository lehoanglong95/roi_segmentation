""" Full assembly of the parts to form the complete network """

from network.unet_3d.unet_parts import *
import torch.nn.functional as F
import torch

class UNet(nn.Module):
    def __init__(self, input_channel, output_channel, soft_dim, is_training, device_1, device_2, trilinear=True):
        super(UNet, self).__init__()
        self.device_1 = device_1
        self.device_2 = device_2
        self.soft_dim = soft_dim
        self.is_training = is_training
        self.trilinear = trilinear

        self.inc = DoubleConv(input_channel, 32).to(self.device_1)
        self.down1 = Down(32, 64).to(self.device_1)
        self.down2 = Down(64, 128).to(self.device_1)
        self.down3 = Down(128, 256).to(self.device_1)
        self.down4 = Down(256, 256).to(self.device_1)
        self.up1 = Up(512, 128, trilinear).to(self.device_2)
        self.up2 = Up(256, 64, trilinear).to(self.device_2)
        self.up3 = Up(128, 32, trilinear).to(self.device_2)
        self.up4 = Up(64, 16, trilinear).to(self.device_2)
        self.outc = OutConv(16, output_channel).to(self.device_2)
        self.soft = nn.Softmax(dim=self.soft_dim).to(self.device_2)

    def forward(self, x):
        x = x.to(self.device_1, dtype=torch.float)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5.to(self.device_2), x4.to(self.device_2))
        x = self.up2(x, x3.to(self.device_2))
        x = self.up3(x, x2.to(self.device_2))
        x = self.up4(x.to(self.device_2), x1.to(self.device_2))
        out = self.outc(x.to(self.device_2))

        if self.is_training:
            return out
        return torch.argmax(self.soft(out), dim=self.soft_dim)

if __name__ == '__main__':
    # a = torch.rand((1, 30, 20, 176, 256))
    # b = torch.rand((1, 31, 20, 176, 256))
    # a = torch.rand((1, 33, 20, 176, 256))
    # a = torch.rand((1, 45, 20, 176, 256))
    from utils.utils import count_parameters
    from datetime import datetime
    a = torch.rand((1, 2, 45, 224, 224)).to(torch.device("cpu"))
    net = UNet(2, 2, 1, True, torch.device("cpu"), torch.device("cpu")).to(torch.device("cpu"))
    t1 = datetime.now()
    b = net(a)
    print((datetime.now() - t1).total_seconds())
    # print(b.shape)
    # print(torch.squeeze(b, dim=2).shape)
    print(count_parameters(net))
    # _p = net(a)
    # print(_p.shape)



