import torch
import torch.nn as nn

import torch
import torch.nn.modules
import torch.nn as nn
from torch.nn import functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        # 1×1 Conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # dilation = 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # dilation = 12
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # dilation = 18
        self.conv18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 每个分支的 Global Pooling
        self.gap_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),

            nn.ReLU(inplace=True)
        )

        # 最终融合
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def global_context(self, x):

        size = x.size()[2:]

        gap = F.adaptive_avg_pool2d(x, 1)
        gap = self.gap_conv(gap)

        gap = F.interpolate(gap, size=size, mode='bilinear', align_corners=True)

        return x + gap


    def forward(self, x):

        x1 = self.conv1(x)

        x2 = self.conv6(x)
        x2 = self.global_context(x2)

        x3 = self.conv12(x)
        x3 = self.global_context(x3)

        x4 = self.conv18(x)
        x4 = self.global_context(x4)

        x = torch.cat([x1, x2, x3, x4], dim=1)

        return self.project(x)


#/////
# class FCANet(nn.Module):
#     def __init__(self, channel, reduction=16, dct_h=7, dct_w=7):
#         super(FCANet, self).__init__()
#
#         self.dct_h = dct_h
#         self.dct_w = dct_w
#
#         self.avgpool = nn.AdaptiveAvgPool2d((dct_h, dct_w))
#
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#
#         b, c, h, w = x.size()
#
#         y = self.avgpool(x)
#
#         # 频域压缩
#         y = torch.mean(y, dim=[2, 3])
#
#         y = self.fc(y).view(b, c, 1, 1)
#
#         return x * y.expand_as(x)
#
# class ASPP(nn.Module):
#     def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
#         super(ASPP, self).__init__()
#         self.branch1 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#         self.branch2 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#         self.branch3 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#         self.branch4 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#         self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
#         self.branch5_relu = nn.ReLU(inplace=True)
#
#         self.conv_cat = nn.Sequential(
#             nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#
#         self.FCANet = FCANet(channel=dim_out * 5, dct_h=42, dct_w=80)
#
#     def forward(self, x):
#         [b, c, row, col] = x.size()
#         conv1x1 = self.branch1(x)
#         conv3x3_1 = self.branch2(x)
#         conv3x3_2 = self.branch3(x)
#         conv3x3_3 = self.branch4(x)
#         global_feature = torch.mean(x, 2, True)
#         global_feature = torch.mean(global_feature, 3, True)
#         global_feature = self.branch5_conv(global_feature)
#         global_feature = self.branch5_relu(global_feature)
#         global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
#
#         feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
#         # 加入频率注意力机制
#         fcanet = self.FCANet(feature_cat)
#         result = self.conv_cat(fcanet)
#         return result

# class ASPP(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ASPP, self).__init__()
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#         self.conv6 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3,
#                       padding=6, dilation=6, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#         self.conv12 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3,
#                       padding=12, dilation=12, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#         self.conv18 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3,
#                       padding=18, dilation=18, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#         self.project = nn.Sequential(
#             nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.conv6(x)
#         x3 = self.conv12(x)
#         x4 = self.conv18(x)
#
#         x = torch.cat([x1, x2, x3, x4], dim=1)
#         return self.project(x)
