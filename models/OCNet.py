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
# Object Context Block
# -----------------------------
class ObjectContextBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.query = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)

        self.softmax = nn.Softmax(dim=-1)

        self.project = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):

        B, C, H, W = x.size()

        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(B, -1, H * W)
        value = self.value(x).view(B, -1, H * W)

        sim_map = torch.bmm(query, key)
        sim_map = self.softmax(sim_map)

        context = torch.bmm(value, sim_map.permute(0, 2, 1))
        context = context.view(B, C, H, W)

        out = self.project(context)

        return out + x


# -----------------------------
# OC Module
# -----------------------------
class OCModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = ConvBNReLU(in_channels, out_channels, 3, 1, 1)

        self.oc_block = ObjectContextBlock(out_channels)

        self.conv2 = ConvBNReLU(out_channels, out_channels, 3, 1, 1)

    def forward(self, x):

        x = self.conv1(x)

        x = self.oc_block(x)

        x = self.conv2(x)

        return x


# -----------------------------
# OCNet
# -----------------------------
class OCNet(nn.Module):

    def __init__(self, n_channels=3, n_classes=4):
        super().__init__()

        backbone = models.resnet50(pretrained=True)

        if n_channels != 3:
            backbone.conv1 = nn.Conv2d(
                n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

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

        # OC module
        self.oc = OCModule(2048, 512)

        # segmentation head
        self.cls = nn.Sequential(
            ConvBNReLU(512, 256, 3, 1, 1),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, n_classes, 1)
        )

    def forward(self, x):

        input_size = x.size()[2:]

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.oc(x)

        x = self.cls(x)

        x = F.interpolate(
            x,
            size=input_size,
            mode='bilinear',
            align_corners=True
        )

        return x