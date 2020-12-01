import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):

    def __init__(self, input_channel, output_channel, soft_dim, is_training):
        super(UNet, self).__init__()

        self.soft_dim = soft_dim
        self.is_training= is_training

        self.encoder1 = nn.Conv2d(input_channel, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.encoder4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.encoder5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)

        self.decoder1 = nn.Conv2d(512, 256, 3, stride=1, padding=1)  # b, 16, 5, 5
        self.decoder2 = nn.Conv2d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 = nn.Conv2d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(32, output_channel, 3, stride=1, padding=1)

        self.soft = nn.Softmax(dim=soft_dim)

    def forward(self, x):
        # x = x.type(torch.FloatTensor).to(torch.device("cuda: 1"))
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out
        out = F.relu(F.max_pool2d(self.encoder4(out), 2, 2))
        t4 = out
        out = F.relu(F.max_pool2d(self.encoder5(out), 2, 2))

        # t2 = out
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        # print(out.shape,t4.shape)
        out = torch.add(out, t4)
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        out = torch.add(out, t3)
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear', align_corners=True))
        # print(out.shape)

        n, c, h, w = out.shape
        out = out.view(n, 2, int(c // 2), h, w)
        if self.is_training:
            return out
        return torch.argmax(self.soft(out), dim=self.soft_dim)
        # out = self.soft(out)
        # out = torch.argmax(out, dim=self.soft_dim)
        # return out[:, :, 1, :, :]
        # return out.type(torch.FloatTensor).to(torch.device("cuda: 1"))

if __name__ == '__main__':
    a = torch.rand((1, 44, 256, 256))
    net = UNet(44, 88, 1, True)
    pred = net(a)
    print(pred.shape)
    # print(pred[:, :, 1, :, :].shape)