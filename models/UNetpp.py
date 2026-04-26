import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# 基础卷积模块
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# ----------------------------
# UNet++ 模型
# ----------------------------
class UNetPP(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, base_ch=32):
        super(UNetPP, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_ch = base_ch

        # 编码器
        self.conv0_0 = ConvBlock(n_channels, base_ch)
        self.conv1_0 = ConvBlock(base_ch, base_ch*2)
        self.conv2_0 = ConvBlock(base_ch*2, base_ch*4)
        self.conv3_0 = ConvBlock(base_ch*4, base_ch*8)
        self.conv4_0 = ConvBlock(base_ch*8, base_ch*16)
        self.pool = nn.MaxPool2d(2)

        # 解码器 / 嵌套
        # convX_Y 输入通道数 = 拼接特征总和
        self.conv3_1 = ConvBlock(base_ch*8 + base_ch*16, base_ch*8)
        self.conv2_1 = ConvBlock(base_ch*4 + base_ch*8, base_ch*4)
        self.conv1_1 = ConvBlock(base_ch*2 + base_ch*4, base_ch*2)
        self.conv0_1 = ConvBlock(base_ch + base_ch*2, base_ch)

        self.conv2_2 = ConvBlock(base_ch*4 + base_ch*4 + base_ch*8, base_ch*4)
        self.conv1_2 = ConvBlock(base_ch*2 + base_ch*2 + base_ch*4, base_ch*2)
        self.conv0_2 = ConvBlock(base_ch + base_ch + base_ch*2, base_ch)

        self.conv1_3 = ConvBlock(base_ch*2 + base_ch*2 + base_ch*2 + base_ch*4, base_ch*2)
        self.conv0_3 = ConvBlock(base_ch + base_ch + base_ch + base_ch*2, base_ch)

        self.conv0_4 = ConvBlock(base_ch + base_ch + base_ch + base_ch + base_ch*2, base_ch)

        # 输出
        self.final = nn.Conv2d(base_ch, n_classes, kernel_size=1)

    # ----------------------------
    # forward
    # ----------------------------
    def forward(self, x):
        # 编码器
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # 解码器 / 上采样 + 拼接，自动对齐尺寸
        x4_up = F.interpolate(x4_0, size=x3_0.shape[2:], mode='bilinear', align_corners=True)
        x3_1 = self.conv3_1(torch.cat([x3_0, x4_up], dim=1))

        x3_up = F.interpolate(x3_1, size=x2_0.shape[2:], mode='bilinear', align_corners=True)
        x2_1 = self.conv2_1(torch.cat([x2_0, x3_up], dim=1))

        x2_up = F.interpolate(x2_1, size=x1_0.shape[2:], mode='bilinear', align_corners=True)
        x1_1 = self.conv1_1(torch.cat([x1_0, x2_up], dim=1))

        x1_up = F.interpolate(x1_1, size=x0_0.shape[2:], mode='bilinear', align_corners=True)
        x0_1 = self.conv0_1(torch.cat([x0_0, x1_up], dim=1))

        # 二级嵌套
        x3_1_up = F.interpolate(x3_1, size=x2_0.shape[2:], mode='bilinear', align_corners=True)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, x3_1_up], dim=1))

        x2_2_up = F.interpolate(x2_2, size=x1_0.shape[2:], mode='bilinear', align_corners=True)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, x2_2_up], dim=1))

        x1_2_up = F.interpolate(x1_2, size=x0_0.shape[2:], mode='bilinear', align_corners=True)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, x1_2_up], dim=1))

        # 三级嵌套
        x2_2_up2 = F.interpolate(x2_2, size=x1_0.shape[2:], mode='bilinear', align_corners=True)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, x2_2_up2], dim=1))

        x1_3_up = F.interpolate(x1_3, size=x0_0.shape[2:], mode='bilinear', align_corners=True)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, x1_3_up], dim=1))

        # 四级嵌套
        x1_3_up2 = F.interpolate(x1_3, size=x0_0.shape[2:], mode='bilinear', align_corners=True)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, x1_3_up2], dim=1))

        output = self.final(x0_4)
        return output

