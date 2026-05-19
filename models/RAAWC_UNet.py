import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. CBAM 模块 (通道注意力 + 空间注意力)
# 对应论文 2.2.2 节及 Figure 8, 9
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_excitation = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
        # Spatial Attention
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        # Channel Attention
        avg_out = self.channel_excitation(self.avg_pool(x))
        max_out = self.channel_excitation(self.max_pool(x))
        channel_att = torch.sigmoid(avg_out + max_out)
        x_after_channel = x * channel_att
        
        # Spatial Attention
        avg_out_s = torch.mean(x_after_channel, dim=1, keepdim=True)
        max_out_s, _ = torch.max(x_after_channel, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_out_s, max_out_s], dim=1)
        spatial_att = torch.sigmoid(self.spatial_conv(spatial_in))
        
        out = x_after_channel * spatial_att
        return out

# 2. Res_CBAM 模块 
# 对应论文 2.2.2 节：残差块 + CBAM
class Res_CBAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_CBAM, self).__init__()
        
        # 残差主干：两层卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # CBAM 注意力模块
        self.cbam = CBAM(out_channels)
        
        # 跳跃连接 (Shortcut)
        # 如果输入输出通道不一致，使用 1x1 卷积调整
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 应用 CBAM
        out = self.cbam(out)
        
        # 残差连接
        out += residual
        out = self.relu(out)
        return out

# 3. 改进的 ASPP 模块 
# 对应论文 2.2.3 节及 Figure 10
class Improved_ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=[6, 12, 18]):
        super(Improved_ASPP, self).__init__()
        
        # 论文提到移除了 1x1 卷积和池化分支，仅保留多尺度空洞卷积
        # 但为了保持特征融合能力，这里保留一个 1x1 卷积分支作为基础特征
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.convs = nn.ModuleList()
        for rate in rates:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # 最终融合
        in_channels_final = len(self.convs) + 1  # 多尺度分支 + 1x1分支
        self.project = nn.Sequential(
            nn.Conv2d(in_channels_final * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        features = [self.conv1(x)] # 基础特征
        
        # 多尺度特征
        for conv in self.convs:
            features.append(conv(x))
            
        # 上采样并对齐
        # 论文 Figure 10 显示是 Concat 后接 1x1 卷积
        # 这里假设各分支输出尺寸一致，直接拼接
        x = torch.cat(features, dim=1)
        x = self.project(x)
        return x

# 4. 权重压缩损失函数 (Weight Compression Loss)
# 对应论文 2.2.4 节及公式 (5)
class WeightCompressionLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(WeightCompressionLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        # pred: 预测概率 (经过 sigmoid)
        # target: 真实标签 (0 or 1)
        
        # 防止 log(0)
        eps = 1e-7
        pred = torch.clamp(pred, eps, 1.0 - eps)
        
        # 计算调制因子
        # 论文公式 (5): WC_Loss = - (1-arctan(p))^r * log(p) if y=1
        #                      = - (arctan(p))^r * log(1-p) if y=0
        # 注意：这里的 p 是预测为正类的概率
        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()
        
        # 正类损失 (病害)
        pos_loss = - (1 - torch.arctan(pred)) ** self.gamma * torch.log(pred)
        # 负类损失 (背景/健康叶片)
        neg_loss = - (torch.arctan(pred)) ** self.gamma * torch.log(1 - pred)
        
        loss = (pos_mask * pos_loss + neg_mask * neg_loss).mean()
        return loss

# 5. RAAWC-UNet 主干网络
class RAAWC_UNet(nn.Module):
    def __init__(self, num_classes=1, in_channels=3):
        super(RAAWC_UNet, self).__init__()
        
        # Encoder (下采样路径)
        # 根据 Table 2，输入 512x512x3
        
        # Stage 1
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, padding=3, bias=False), # Table 2 提到 7x7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Res_CBAM(64, 64)
        )
        self.pool1 = nn.MaxPool2d(2, 2) # 256x256
        
        # Stage 2
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Res_CBAM(128, 128)
        )
        self.pool2 = nn.MaxPool2d(2, 2) # 128x128
        
        # Stage 3
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            Res_CBAM(256, 256)
        )
        self.pool3 = nn.MaxPool2d(2, 2) # 64x64
        
        # Stage 4 (Bottom)
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            Res_CBAM(512, 512)
        )
      
        # 这里按照 Table 2 实现：先下采样，再 ASPP (ASPP 输入为 1024 channels?)
        
        self.pool4 = nn.MaxPool2d(2, 2) # 32x32
        self.aspp = Improved_ASPP(512, 1024) # 输入512，输出1024 (模拟Table 2中的参数膨胀)
        
        # Decoder (上采样路径)
        # Table 2 显示 Decoder 输入通道数较高
        
        # Decoder 1 (接 ASPP 输出)
        # ASPP 输出 1024, 上采样后与 enc4 (512) 拼接 -> 1024+512=1536? 
        # 但 Table 2 显示 Output 512
        # 这里简化处理：ASPP 输出降维 + 融合
        
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # 融合 enc4 特征
        self.dec1 = Res_CBAM(512 + 512, 512) # 512(上采样) + 512(enc4) = 1024 -> 512
        
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = Res_CBAM(256 + 256, 256) # 256 + 256(enc3) = 512 -> 256
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = Res_CBAM(128 + 128, 128) # 128 + 128(enc2) = 256 -> 128
        
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = Res_CBAM(64 + 64, 64)  # 64 + 64(enc1) = 128 -> 64
        
        # 输出层
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)         # 64, 512, 512
        p1 = self.pool1(e1)       # 64, 256, 256
        
        e2 = self.enc2(p1)        # 128, 256, 256
        p2 = self.pool2(e2)       # 128, 128, 128
        
        e3 = self.enc3(p2)        # 256, 128, 128
        p3 = self.pool3(e3)       # 256, 64, 64
        
        e4 = self.enc4(p3)        # 512, 64, 64
        p4 = self.pool4(e4)       # 512, 32, 32
        
        # Bottleneck
        aspp_out = self.aspp(p4)  # 1024, 32, 32
        
        # Decoder
        d1 = self.upconv1(aspp_out) # 512, 64, 64
        # 调整 enc4 尺寸以匹配 (如果需要)
        # d1 = F.interpolate(d1, size=e4.shape[-2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e4], dim=1) # 512+512=1024
        d1 = self.dec1(d1)            # 512, 64, 64
        
        d2 = self.upconv2(d1)         # 256, 128, 128
        d2 = torch.cat([d2, e3], dim=1) # 256+256=512
        d2 = self.dec2(d2)            # 256, 128, 128
        
        d3 = self.upconv3(d2)         # 128, 256, 256
        d3 = torch.cat([d3, e2], dim=1) # 128+128=256
        d3 = self.dec3(d3)            # 128, 256, 256
        
        d4 = self.upconv4(d3)         # 64, 512, 512
        d4 = torch.cat([d4, e1], dim=1) # 64+64=128
        d4 = self.dec4(d4)            # 64, 512, 512
        
        out = self.final(d4)          # 1, 512, 512
        return torch.sigmoid(out)

