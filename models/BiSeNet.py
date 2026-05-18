import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# -----------------------------
# Basic Conv Block
# -----------------------------
class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_chan, out_chan, ks, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


# -----------------------------
# Spatial Path
# -----------------------------
class SpatialPath(nn.Module):

    def __init__(self, in_channels=3):
        super().__init__()

        self.conv1 = ConvBNReLU(in_channels, 64, 7, 2, 3)
        self.conv2 = ConvBNReLU(64, 128, 3, 2, 1)
        self.conv3 = ConvBNReLU(128, 256, 3, 2, 1)
        self.conv4 = ConvBNReLU(256, 256, 1, 1, 0)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x


# -----------------------------
# Attention Refinement Module
# -----------------------------
class ARM(nn.Module):

    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.conv = ConvBNReLU(in_chan, out_chan, 3, 1, 1)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_chan, out_chan, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        feat = self.conv(x)

        atten = self.attention(feat)

        out = feat * atten

        return out


# -----------------------------
# Context Path
# -----------------------------
class ContextPath(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = models.resnet18(pretrained=True)

        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.arm16 = ARM(256, 256)
        self.arm32 = ARM(512, 256)

    def forward(self, x):

        x = self.layer0(x)
        x = self.layer1(x)

        feat8 = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)

        feat32 = self.arm32(feat32)
        feat32_up = F.interpolate(feat32, size=feat16.shape[2:],
                                  mode='bilinear', align_corners=True)

        feat16 = self.arm16(feat16)
        feat16 = feat16 + feat32_up

        feat16_up = F.interpolate(feat16, size=feat8.shape[2:],
                                  mode='bilinear', align_corners=True)
        return feat16_up


# -----------------------------
# Feature Fusion Module
# -----------------------------
class FeatureFusionModule(nn.Module):

    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.conv = ConvBNReLU(in_chan, out_chan, 1, 1, 0)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_chan, out_chan // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan // 4, out_chan, 1),
            nn.Sigmoid()
        )

    def forward(self, sp, cp):

        feat = torch.cat([sp, cp], dim=1)

        feat = self.conv(feat)

        atten = self.attention(feat)

        feat = feat * atten + feat

        return feat


# -----------------------------
# BiSeNet
# -----------------------------
class BiSeNet(nn.Module):

    def __init__(self, n_channels=3, n_classes=4):
        super().__init__()

        self.spatial_path = SpatialPath(n_channels)

        self.context_path = ContextPath()

        self.ffm = FeatureFusionModule(512, 256)

        self.classifier = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):

        input_size = x.size()[2:]

        sp = self.spatial_path(x)

        cp = self.context_path(x)

        feat = self.ffm(sp, cp)

        out = self.classifier(feat)

        out = F.interpolate(out, size=input_size,
                            mode='bilinear', align_corners=True)

        return out