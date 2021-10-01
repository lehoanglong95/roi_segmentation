# import torch
# import torch.nn as nn
# from network.efficientnet_pytorch import EfficientNet
#
# class SqueezeExciteBlock(nn.Module):
#
#     def __init__(self, in_channels, ratio=8):
#         super().__init__()
#         self.dense1 = nn.Linear(in_channels, in_channels // ratio, bias=False)
#         self.relu1 = nn.ReLU()
#         self.dense2 = nn.Linear(in_channels // ratio, in_channels, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         init = x
#         width_axis = 2
#         height_axis = 3
#
#         se = nn.AvgPool2d(kernel_size=(init.shape[width_axis], init.shape[height_axis]))(init)
#         se = torch.unsqueeze(se, dim=2)
#         se = torch.unsqueeze(se, dim=3)
#         se = self.relu1(self.dense1(se))
#         se = self.sigmoid(self.dense2(se))
#
#         return torch.mul(init, se)
#
# class ConvBlock(nn.Module):
#
#     def __init__(self, in_channels):
#         self.conv_1 = nn.Conv2d(in_channels, (3, 3))
#         self.batch_norm1 = nn.BatchNorm2d()
#         self.relu1 = nn.ReLU()
#
#         self.conv_2 = nn.Conv2d(in_channels, (3, 3))
#         self.batch_norm2 = nn.BatchNorm2d()
#         self.relu2 = nn.ReLU()
#
#         self.squeeze_excite_block = SqueezeExciteBlock(in_channels)
#
#     def forward(self, x):
#         x = self.conv_1(x)
#         x = self.batch_norm1(x)
#         x = self.relu1(x)
#
#         x = self.conv_2(x)
#         x = self.batch_norm2(x)
#         x = self.relu2(x)
#
#         x = self.squeeze_excite_block(x)
#
#         return x
#
# class Encoder1(nn.Module):
#
#     def __init__(self, in_channels, model_name="efficientnet-b5"):
#         super().__init__()
#         self.efficient_net = EfficientNet.from_pretrained(model_name, num_classes=1, in_channels=in_channels)
#
#     def forward(self, x):
#         end_points = self.efficient_net.extract_endpoints(x)
#         return end_points["reduction_5"], [end_points["reduction_1"], end_points["reduction_2"],
#                                            end_points["reduction_3"], end_points["reduction_4"]]
#
# class Decoder1(nn.Module)
#
#     def __init__(self, num_filters=[]):
#         self.upsample_1 = nn.UpsamplingBilinear2d(size=(2, 2), scale_factor=2)
#         self.conv_block_1 = ConvBlock(num_filters[0])
#         self.upsample_2 = nn.UpsamplingBilinear2d(size=(2, 2), scale_factor=2)
#         self.conv_block_2 = ConvBlock(num_filters[1])
#         self.upsample_3 = nn.UpsamplingBilinear2d(size=(2, 2), scale_factor=2)
#         self.conv_block_3 = ConvBlock(num_filters[2])
#         self.upsample_4 = nn.UpsamplingBilinear2d(size=(2,2), scale_factor=2)
#         self.conv_block_4 = ConvBlock(num_filters[3])
#
#     def forward(self, x, skip_1):
#         x = self.upsample_1(x)
#         x = torch.cat([x, skip_1[0]])
#         x = self.conv_block_1(x)
#         x = self.upsample_2(x)
#         x = torch.cat([x, skip_1[1]])
#         x = self.conv_block_2(x)
#         x = self.upsample_3(x)
#         x = torch.cat([x, skip_1[2]])
#         x = self.conv_block_3(x)
#         x = self.upsample_4(x)
#         x = torch.cat([x, skip_1[3]])
#         x = self.conv_block_4(x)
#         return x
#
# class Encoder2(nn.Module):
#
#     def __init__(self, num_filters=[]):
#         super().__init__()
#         self.conv_block_1 = ConvBlock(num_filters[0])
#         self.conv_block_2 = ConvBlock(num_filters[1])
#         self.conv_block_3 = ConvBlock(num_filters[2])
#         self.conv_block_4 = ConvBlock(num_filters[3])
#         self.max_pool = nn.MaxPool2d()
#
#     def forward(self, x):
#         skip_connections = []
#         x = self.conv_block_1(x)
#         skip_connections.append(x)
#         x = self.max_pool(x)
#         x = self.conv_block_2(x)
#         skip_connections.append(x)
#         x = self.max_pool(x)
#         x = self.conv_block_3(x)
#         skip_connections.append(x)
#         x = self.max_pool(x)
#         x = self.conv_block_4(x)
#         skip_connections.append(x)
#         x = self.max_pool(x)
#         return x, skip_connections
#
# class Decoder2(nn.Module)
#
#     def __init__(self, num_filters=[]):
#         self.upsample_1 = nn.UpsamplingBilinear2d(size=(2, 2), scale_factor=2)
#         self.conv_block_1 = ConvBlock(num_filters[0])
#         self.upsample_2 = nn.UpsamplingBilinear2d(size=(2, 2), scale_factor=2)
#         self.conv_block_2 = ConvBlock(num_filters[1])
#         self.upsample_3 = nn.UpsamplingBilinear2d(size=(2, 2), scale_factor=2)
#         self.conv_block_3 = ConvBlock(num_filters[2])
#         self.upsample_4 = nn.UpsamplingBilinear2d(size=(2, 2), scale_factor=2)
#         self.conv_block_4 = ConvBlock(num_filters[3])
#
#     def __int__(self, x, skip_1, skip_2):
#         x = self.upsample_1(x)
#         x = torch.cat([x, skip_1[0], skip_2[0]])
#         x = self.conv_block_1(x)
#         x = self.upsample_2(x)
#         x = torch.cat([x, skip_1[1], skip_2[1]])
#         x = self.conv_block_2(x)
#         x = self.upsample_3(x)
#         x = torch.cat([x, skip_1[2], skip_2[2]])
#         x = self.conv_block_3(x)
#         x = self.upsample_4(x)
#         x = torch.cat([x, skip_1[3], skip_2[3]])
#         x = self.conv_block_5(x)
#         return x
#
# class ASPP(nn.Module):
#
#     def __init__(self, in_channels):
#         self.conv_1 = nn.Conv2d(in_channels, in_channels, 1)
#         self.batchnorm_1 = nn.BatchNorm2d()
#         self.relu_1 = nn.ReLU()
#         self.upsample_1 = nn.UpsamplingBilinear2d()
#
#         self.conv_2 = nn.Conv2d(in_channels, in_channels, 1, dilation=1, bias=False)
#         self.batchnorm_2 = nn.BatchNorm2d()
#         self.relu_2 = nn.ReLU()
#
#         self.conv_3 = nn.Conv2d(in_channels, in_channels, 3, dilation=6, padding=6, bias=False)
#         self.batchnorm_3 = nn.BatchNorm2d()
#         self.relu_3 = nn.ReLU()
#
#         self.conv_4 = nn.Conv2d(in_channels, in_channels, 3, dilation=12, padding=12, bias=False)
#         self.batchnorm_4 = nn.BatchNorm2d()
#         self.relu_4 = nn.ReLU()
#
#         self.conv_5 = nn.Conv2d(in_channels, in_channels, 3, dilation=18, padding=18, bias=False)
#         self.batchnorm_5 = nn.BatchNorm2d()
#         self.relu_5 = nn.ReLU()
#
#         self.conv_6 = nn.Conv2d(in_channels, in_channels, 1, dilation=1, bias=False)
#         self.batchnorm_6 = nn.BatchNorm2d()
#         self.relu_6 = nn.ReLU()
#
#     def forward(self, x):
#         width = x.shape[1]
#         height = x.shape[2]
#         x1 = nn.AvgPool2d(kernel_size=(width, height))(x)
#         x1 = self.conv_1(x1)
#         x1 = self.batchnorm_1(x1)
#         x1 = self.relu_1(x1)
#         x1 = self.upsample_1(x1)
#
#         x2 = self.conv_2(x)
#         x2 = self.batchnorm_2(x2)
#         x2 = self.relu_2(x2)
#
#         x3 = self.conv_3(x)
#         x3 = self.batchnorm_3(x3)
#         x3 = self.relu_3(x3)
#
#         x4 = self.conv_4(x)
#         x4 = self.batchnorm_4(x4)
#         x4 = self.relu_4(x4)
#
#         x5 = self.conv_5(x)
#         x5 = self.batchnorm_5(x5)
#         x5 = self.relu_5(x5)
#
#         x = torch.cat([x1, x2, x3, x4, x5])
#
#         x = self.conv_6(x)
#         x = self.batchnorm_6(x)
#         x = self.relu_6(x)
#
#         return x
#
#
#
#
