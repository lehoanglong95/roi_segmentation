""" Full assembly of the parts to form the complete network """

from network.unet_3d.unet_parts import *
import torch.nn.functional as F
import torch

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, soft_dim, is_training, device, trilinear=True):
        super(UNet, self).__init__()
        # self.n_channels = n_channels
        self.n_classes = n_classes
        self.device = device
        self.soft_dim = soft_dim
        self.is_training = is_training
        self.trilinear = trilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if trilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, trilinear)
        self.up2 = Up(512, 256 // factor, trilinear)
        self.up3 = Up(256, 128 // factor, trilinear)
        self.up4 = Up(128, 64, trilinear)
        self.outc = OutConv(64, n_classes)
        self.soft = nn.Softmax(dim=self.soft_dim)

    def forward(self, x):
        if self.device:
            x = x.to(self.device, dtype=torch.float)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        n, c, d, h, w = logits.shape
        out = logits.view(n, 2, int(c // 2), d, h, w)
        if self.is_training:
            return out
        return torch.argmax(self.soft(out), dim=self.soft_dim)

if __name__ == '__main__':
    # a = torch.rand((1, 30, 20, 176, 256))
    # b = torch.rand((1, 31, 20, 176, 256))
    # a = torch.rand((1, 33, 20, 176, 256))
    # a = torch.rand((1, 45, 20, 176, 256))
    from utils.utils import count_parameters
    a = torch.rand((1, 1, 50, 300, 300)).to(torch.device("cuda: 1"))
    net = UNet(1, 2, 1, True, device=None).to(torch.device("cuda: 1"))
    print(net(a).shape)
    # print(count_parameters(net))
    # _p = net(a)
    # print(_p.shape)



