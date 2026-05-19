import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 基础卷积块 (参考图2-d Feature Extraction Block)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# 2. 前景注意力块 (FAB) - 核心组件 (参考图2-c)
# 论文公式(1)和(2)的实现逻辑
class ForegroundAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ForegroundAttentionBlock, self).__init__()
        
        # Channel-wise Weighting (Global Avg Pooling + Sigmoid)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial-wise Weighting (Local Max Pooling + Sigmoid)
        # 论文提及 Local Max Pooling，这里使用 7x7 模拟局部感受野
        self.max_pool = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.conv_sp = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        channel_att = self.ca(self.gap(x))
        x_channel = x * channel_att
        
        # Spatial Attention
        spatial_att = self.conv_sp(self.max_pool(x_channel))
        
        return spatial_att # 输出 A^f (前景注意力图)

# 3. 三重注意力模块 (TAM) - (参考图2-a)
class TripleAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(TAM, self).__init__()
        self.fab = ForegroundAttentionBlock(in_channels)

    def forward(self, x):
        # A^f: 前景/缺陷注意力
        A_f = self.fab(x)
        
        # 论文公式 (1): A^b = 1 - (|A^f - 0.5| / 0.5)
        # 计算边界注意力 A^b
        A_b = 1 - (torch.abs(A_f - 0.5) / 0.5)
        
        # 论文公式 (2): A^r = 1 - A^f
        # 计算反向/背景注意力 A^r
        A_r = 1 - A_f
        
        return A_f, A_r, A_b 

# 4. 注意力融合模块 (AFM) - (参考图2-b)
class AttentionFusionModule(nn.Module):
    def __init__(self, in_channels):
        super(AFusionModule, self).__init__()
        # 用于处理解码器输入 X1
        self.conv1 = ConvBlock(in_channels * 2, in_channels) # 拼接了 X1 和 权重
        # 用于处理编码器输入 X2/X3
        self.conv2 = ConvBlock(in_channels, in_channels)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_decoder, x_encoder, A_f, A_r):
        # x_decoder (D_i): 解码器特征
        # x_encoder: 编码器特征
        
        # 论文描述
        weighted_encoder = x_encoder * A_f
        weighted_decoder = x_decoder * A_r
        
        # 沿通道方向拼接
       
        fused = torch.cat([weighted_decoder, self.conv2(weighted_encoder)], dim=1)
        
        # 生成注意力增强的预测图
        out = self.final_conv(fused)
        return out

# 5. TAG-Net 主干网络
class TAGNet(nn.Module):
    def __init__(self, num_classes=1, in_channels=3):
        super(TAGNet, self).__init__()
        
        # 编码器
        self.encoder1 = ConvBlock(in_channels, 64)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256)
        self.encoder4 = ConvBlock(256, 512)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # 解码器
        # 上采样层
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        # 解码器融合块 (包含 AFM)
        self.decoder3 = ConvBlock(256 + 256, 256) # 通道数根据拼接调整
        self.decoder2 = ConvBlock(128 + 128, 128)
        self.decoder1 = ConvBlock(64 + 64, 64)
        
        # 输出层
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # 边界预测输出 (深监督用)
        self.final_boundary = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # 编码器路径
        e1 = self.encoder1(x)       # 64
        p1 = self.pool(e1)
        
        e2 = self.encoder2(p1)     # 128
        p2 = self.pool(e2)
        
        e3 = self.encoder3(p2)     # 256
        p3 = self.pool(e3)
        
        e4 = self.encoder4(p3)     # 512
        
        # 解码器路径 + 注意力机制
        # 第4层 (最深层)
        d4 = self.upconv4(e4)      # 256
        
        # 获取注意力图 (输入为编码器输出和解码器输入的融合？)
      
        A_f, A_r, A_b = TripleAttentionModule(e3.size(1)).to(x.device)(e3)
        
        # AFM 融合
        # d4 与 e3 拼接前先进行注意力加权
        # 论文 AFM 输入: X1 (Decoder), X2/X3 (Encoder)
        fused3 = AttentionFusionModule(e3.size(1))(
            x_decoder=d4, 
            x_encoder=e3, 
            A_f=A_f, 
            A_r=A_r
        )
        d3 = self.decoder3(fused3)
        
        # 第3层
        d3_up = self.upconv3(d3)
        A_f2, A_r2, A_b2 = TripleAttentionModule(e2.size(1)).to(x.device)(e2)
        fused2 = AttentionFusionModule(e2.size(1))(
            x_decoder=d3_up, 
            x_encoder=e2, 
            A_f=A_f2, 
            A_r=A_r2
        )
        d2 = self.decoder2(fused2)
        
        # 第2层
        d2_up = self.upconv2(d2)
        A_f3, A_r3, A_b3 = TripleAttentionModule(e1.size(1)).to(x.device)(e1)
        fused1 = AttentionFusionModule(e1.size(1))(
            x_decoder=d2_up, 
            x_encoder=e1, 
            A_f=A_f3, 
            A_r=A_r3
        )
        d1 = self.decoder1(fused1)
        
        # 输出预测
        pred = self.final(d1)
        boundary_pred = self.final_boundary(d1)
        
        return torch.sigmoid(pred), torch.sigmoid(boundary_pred), A_b, A_b2, A_b3

# --- 损失函数定义 (参考论文公式 5, 6, 7) ---

class TAGNetLoss(nn.Module):
    def __init__(self):
        super(TAGNetLoss, self).__init__()

    def dice_loss(self, pred, target):
        smooth = 1e-5
        intersection = (pred * target).sum()
        return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

    def forward(self, pred, boundary_pred, target, boundary_target):
        # 分割损失 (Dice Loss)
        loss_seg = self.dice_loss(pred, target)
        
        # 边界损失 (Binary Cross Entropy)
      
        loss_bound = F.binary_cross_entropy(boundary_pred, boundary_target)
        
       
        # 论文提到线性调整系数 alpha，此处简化为 0.5
        alpha = 0.5
        total_loss = (1 - alpha) * loss_seg + alpha * loss_bound
        
        return total_loss

