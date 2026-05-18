import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# -----------------------------
# ASPP模块
# -----------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):

        size = x.shape[2:]

        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat4 = self.branch4(x)

        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=True)

        x = torch.cat([feat1, feat2, feat3, feat4, global_feat], dim=1)

        x = self.conv(x)

        return x


# -----------------------------
# Decoder模块
# -----------------------------
class Decoder(nn.Module):

    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()

        self.conv_low = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x, low_level):

        low_level = self.conv_low(low_level)

        x = F.interpolate(x, size=low_level.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x, low_level), dim=1)

        x = self.last_conv(x)

        return x


# -----------------------------
# DeepLabV3+ 主网络
# -----------------------------
class DeepLabV3Plus(nn.Module):

    def __init__(self, n_channels=3, n_classes=4):
        super(DeepLabV3Plus, self).__init__()

        resnet = models.resnet50(pretrained=True)

        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )

        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.aspp = ASPP(2048)

        self.decoder = Decoder(256, n_classes)

    def forward(self, x):

        size = x.shape[2:]

        x = self.layer0(x)
        x = self.maxpool(x)

        low_level = self.layer1(x)

        x = self.layer2(low_level)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.aspp(x)

        x = self.decoder(x, low_level)

        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)

        return x