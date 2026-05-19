import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Squeeze-and-Excitation (SE) 模块
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 2. 改进的 ASPP (Atrous Spatial Pyramid Pooling) 模块
class ImprovedASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=[4, 8, 12, 16]):
        super(ImprovedASPP, self).__init__()
        
        # 1x1 卷积分支
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 多尺度膨胀卷积分支
        self.convs = nn.ModuleList()
        for rate in atrous_rates:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # 全局平均池化分支
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 最终融合卷积
        # 输入通道数 = 1x1卷积 + 4个膨胀卷积 + 全局池化 = 5 * out_channels
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) + 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[-2:]
        res = [self.conv1(x)] # 1x1 卷积结果
        
        # 添加多尺度膨胀卷积结果
        for conv in self.convs:
            res.append(conv(x))
            
        # 添加全局池化结果
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=True)
        res.append(global_feat)
        
        # 拼接所有特征
        x = torch.cat(res, dim=1)
        return self.project(x)

# 3. 主干网络 (Encoder)
class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()
        # 这里简化表示，实际应使用 ResNet-50 的 conv1-conv4
        # 为了演示，构建一个包含 SE 模块的简单下采样路径
        self.stage1 = self._make_layer(in_channels, 64)
        self.stage2 = self._make_layer(64, 128, stride=2)
        self.stage3 = self._make_layer(128, 256, stride=2)
        self.stage4 = self._make_layer(256, 512, stride=2)
        
        # 在 Encoder 输出处加入 SE 模块
        self.se = SEBlock(512)

    def _make_layer(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2apter(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.stage1(x)   # 低级特征 (用于 Decoder 跳跃连接)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)  # 高级特征
        x4 = self.se(x4)      # 应用 SE 模块
        return x4, x1 # 返回高级特征和低级特征

# 4. Decoder (解码器)

class Decoder(nn.Module):
    def __init__(self, num_classes=1, low_level_channels=64):
        super(Decoder, self).__init__()
        
        # 降低低级特征的通道数 (从 64 降到 48，这是 DeepLabv3+ 的标准做法)
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # 上采样后的融合卷积
        self.last_conv = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x, low_level_feat):
        # x 是 ASPP 的输出 (256通道)
        # 上采样 x 到 low_level_feat 的尺寸
        x = F.interpolate(x, size=low_level_feat.shape[-2:], mode='bilinear', align_corners=True)
        
        # 处理低级特征
        low_level_feat = self.conv_low(low_level_feat)
        
        # 拼接
        x = torch.cat((x, low_level_feat), dim=1)
        
        # 最终预测
        x = self.last_conv(x)
        # 上采样到原始输入尺寸
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        
        return x

# 5. 最终模型整合
class ImprovedDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1, in_channels=3):
        super(ImprovedDeepLabV3Plus, self).__init__()
        self.encoder = Encoder(in_channels)
        self.aspp = ImprovedASPP(in_channels=512) # 假设 Encoder 输出 512 通道
        self.decoder = Decoder(num_classes, low_level_channels=64)

    def forward(self, x):
        # 获取高级特征 (用于 ASPP) 和低级特征 (用于 Decoder)
        high_level_feat, low_level_feat = self.encoder(x)
        
        # ASPP 处理
        x = self.aspp(high_level_feat)
        
        # Decoder 恢复分辨率
        x = self.decoder(x, low_level_feat)
        
        return torch.sigmoid(x) # 二分类分割，使用 Sigmoid

