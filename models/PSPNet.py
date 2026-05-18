import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PPM(nn.Module):
    """
    Pyramid Pooling Module
    """
    def __init__(self, in_channels, reduction_channels, bins=(1, 2, 3, 6)):
        super(PPM, self).__init__()

        self.features = nn.ModuleList()

        for bin in bins:
            self.features.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin),
                    nn.Conv2d(in_channels, reduction_channels, 1, bias=False),

                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, x):

        h, w = x.shape[2:]
        pyramids = [x]

        for f in self.features:
            out = f(x)
            out = F.interpolate(out, size=(h, w),
                                mode='bilinear', align_corners=True)
            pyramids.append(out)

        return torch.cat(pyramids, dim=1)


class PSPNet(nn.Module):

    def __init__(self, n_channels=3, n_classes=4, backbone='resnet50'):
        super(PSPNet, self).__init__()

        # ---------------------------
        # Backbone (ResNet)
        # ---------------------------
        resnet = models.resnet50(pretrained=True)

        if n_channels != 3:
            resnet.conv1 = nn.Conv2d(
                n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 输出通道
        fea_dim = 2048

        # ---------------------------
        # Pyramid Pooling Module
        # ---------------------------
        self.ppm = PPM(fea_dim, fea_dim // 4)

        ppm_out_dim = fea_dim + 4 * (fea_dim // 4)

        # ---------------------------
        # Final classifier
        # ---------------------------
        self.classifier = nn.Sequential(
            nn.Conv2d(ppm_out_dim, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Dropout2d(0.1),

            nn.Conv2d(512, n_classes, 1)
        )

    def forward(self, x):

        input_size = x.size()[2:]

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.ppm(x)

        x = self.classifier(x)

        x = F.interpolate(x, size=input_size,
                          mode='bilinear', align_corners=True)

        return x