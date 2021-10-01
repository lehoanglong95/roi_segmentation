""" Full assembly of the parts to form the complete network """

from network.unet_3d.unet_parts import *
import torch.nn.functional as F
import torch

class DoubleUnet(nn.Module):
    def __init__(self, input_channel, output_channel, trilinear=True):
        super(DoubleUnet, self).__init__()
        self.trilinear = trilinear

        self.inc_1 = DoubleConv(input_channel, 32)
        self.down_1_1 = Down(32, 64)
        self.down_1_2 = Down(64, 128)
        self.down_1_3 = Down(128, 256)
        self.down_1_4 = Down(256, 512)
        self.aspp_1 = ASPP(512, 64, (1, 16, 16))
        self.up_1_1 = Up(320, 128, trilinear)
        self.up_1_2 = Up(256, 64, trilinear)
        self.up_1_3 = Up(128, 32, trilinear)
        self.up_1_4 = Up(64, 16, trilinear)

        self.inc_2 = DoubleConv(input_channel, 32)
        self.down_2_1 = Down(32, 64)
        self.down_2_2 = Down(64, 128)
        self.down_2_3 = Down(128, 256)
        self.down_2_4 = Down(256, 512)
        self.aspp_2 = ASPP(512, 64, (1, 16, 16))
        self.up_2_1 = Up2(576, 128, trilinear)
        self.up_2_2 = Up2(384, 64, trilinear)
        self.up_2_3 = Up2(192, 32, trilinear)
        self.up_2_4 = Up2(96, 16, trilinear)

        self.outc = OutConv(16, output_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_1_1 = self.inc_1(x)
        x_1_2 = self.down_1_1(x_1_1)
        x_1_3 = self.down_1_2(x_1_2)
        x_1_4 = self.down_1_3(x_1_3)
        x_1_5 = self.down_1_4(x_1_4)
        x_1_6 = self.aspp_1(x_1_5)

        temp_x = self.up_1_1(x_1_6, x_1_4)
        temp_x = self.up_1_2(temp_x, x_1_3)
        temp_x = self.up_1_3(temp_x, x_1_2)
        temp_x = self.up_1_4(temp_x, x_1_1)
        out_1 = self.sigmoid(self.outc(temp_x))

        x = out_1 * x

        x_2_1 = self.inc_2(x)
        x_2_2 = self.down_2_1(x_2_1)
        x_2_3 = self.down_2_2(x_2_2)
        x_2_4 = self.down_2_3(x_2_3)
        x_2_5 = self.down_2_4(x_2_4)
        x_2_6 = self.aspp_2(x_2_5)
        temp_x = self.up_2_1(x_2_6, x_1_4, x_2_4)
        temp_x = self.up_2_2(temp_x, x_1_3, x_2_3)
        temp_x = self.up_2_3(temp_x, x_1_2, x_2_2)
        temp_x = self.up_2_4(temp_x, x_1_1, x_2_1)
        out_2 = self.sigmoid(self.outc(temp_x))

        return torch.cat([out_1, out_2], dim=1)




if __name__ == '__main__':
    from utils.utils import count_parameters
    from datetime import datetime
    a = torch.rand((1, 2, 45, 224, 224))
    net = DoubleUnet(2, 1)
    t1 = datetime.now()
    b = net(a)
    print((datetime.now() - t1).total_seconds())
    # print(b.shape)
    print(count_parameters(net))



