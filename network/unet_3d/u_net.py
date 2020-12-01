""" Full assembly of the parts to form the complete network """

from network.unet_3d.unet_parts import *
import torch.nn.functional as F
import torch

class UNet(nn.Module):
    def __init__(self, input_channel, output_channel, soft_dim, is_training, device, trilinear=True):
        super(UNet, self).__init__()
        self.device = device
        self.soft_dim = soft_dim
        self.is_training = is_training
        self.trilinear = trilinear

        self.encoder1 = nn.Conv3d(input_channel, 32, 3, padding=1)  # b, 16, 10, 10
        self.encoder2 = nn.Conv3d(32, 64, 3, padding=1)  # b, 8, 3, 3
        self.encoder3 = nn.Conv3d(64, 128, 3, padding=1)
        self.encoder4 = nn.Conv3d(128, 256, 3, padding=1)
        self.encoder5 = nn.Conv3d(256, 512, 3, padding=1)

        self.decoder1 = nn.Conv3d(512, 256, 3, padding=1)  # b, 16, 5, 5
        self.decoder2 = nn.Conv3d(256, 128, 3, padding=1)  # b, 8, 15, 1
        self.decoder3 = nn.Conv3d(128, 64, 3, padding=1)  # b, 1, 28, 28
        self.decoder4 = nn.Conv3d(64, 32, 3, padding=1)
        self.decoder5 = nn.Conv3d(32, output_channel, 3, padding=1)
        self.soft = nn.Softmax(dim=self.soft_dim)

    def forward(self, x):
        if self.device:
            x = x.to(self.device, dtype=torch.float)
        out = F.relu(F.max_pool3d(self.encoder1(x), kernel_size=3, stride=(1, 2, 2), padding=(1, 1, 1)))
        t1 = out
        out = F.relu(F.max_pool3d(self.encoder2(out), kernel_size=3, stride=(1, 2, 2), padding=(1, 1, 1)))
        t2 = out
        out = F.relu(F.max_pool3d(self.encoder3(out), kernel_size=3, stride=(1, 2, 2), padding=(1, 1, 1)))
        t3 = out
        out = F.relu(F.max_pool3d(self.encoder4(out), kernel_size=3, stride=(1, 2, 2), padding=(1, 1, 1)))
        t4 = out
        out = F.relu(F.max_pool3d(self.encoder5(out), kernel_size=3, stride=(1, 2, 2), padding=(1, 1, 1)))

        # t2 = out
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(1, 2, 2), mode='trilinear', align_corners=True))
        out = torch.add(out, t4)
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(1, 2, 2), mode='trilinear', align_corners=True))
        out = torch.add(out, t3)
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(1, 2, 2), mode='trilinear', align_corners=True))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(1, 2, 2), mode='trilinear', align_corners=True))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(1, 2, 2), mode='trilinear', align_corners=True))

        if self.is_training:
            return out
        return torch.argmax(self.soft(out), dim=self.soft_dim)

if __name__ == '__main__':
    # a = torch.rand((1, 30, 20, 176, 256))
    # b = torch.rand((1, 31, 20, 176, 256))
    # a = torch.rand((1, 33, 20, 176, 256))
    # a = torch.rand((1, 45, 20, 176, 256))
    from utils.utils import count_parameters
    a = torch.rand((1, 1, 50, 256, 256)).to(torch.device("cuda: 1"))
    net = UNet(1, 2, 1, True, device=None).to(torch.device("cuda: 1"))
    # b = net(a)
    # print(b.shape)
    # print(torch.squeeze(b, dim=2).shape)
    print(count_parameters(net))
    # _p = net(a)
    # print(_p.shape)



