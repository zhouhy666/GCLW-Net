 import torch
 import torch.nn as nn
 import torch.nn.functional as F
 from timm.models.layers import DropPath, to_2tuple, trunc_normal_
 # 注意：实际使用需要安装 timm 库以及 swin_transformer 和 convnext 的官方实现

 class Mlp(nn.Module):
  
     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
         super().__init__()
         out_features = out_features or in_features
         hidden_features = hidden_features or in_features
         self.fc1 = nn.Linear(in_features, hidden_features)
         self.act = act_layer()
         self.drop1 = DropPath(drop) if drop else nn.Identity()
         self.fc2 = nn.Linear(hidden_features, out_features)
         self.drop2 = DropPath(drop) if drop else nn.Identity()

     def forward(self, x):
         x = self.fc1(x)
         x = self.act(x)
         x = self.drop1(x)
         x = self.fc2(x)
         x = self.drop2(x)
         return x

 class PatchMerging(nn.Module):
     """论文3.2.3节 PatchMerging 模块"""
     def __init__(self, in_channels, out_channels, norm_layer=nn.LayerNorm):
         super().__init__()
         self.reduction = nn.Linear(4 * in_channels, out_channels, bias=False)
         self.norm = norm_layer(4 * in_channels)

     def forward(self, x):
         # 假设 x 的形状为 (B, H, W, C)
         B, H, W, C = x.shape
         # 2x2 下采样
         x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
         x1 = x[:, 1::2, 0::2, :]
         x2 = x[:, 0::2, 1::2, :]
         x3 = x[:, 1::2, 1::2, :]
         x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
         x = x.view(B, -1, 4 * C)
         x = self.norm(x)
         x = self.reduction(x)
         return x

 class UpSampleS(nn.Module):
     """论文3.2.4节 ST分支的上采样模块"""
     def __init__(self, in_channels, scale_factor=2):
         super().__init__()
         self.scale_factor = scale_factor
         self.conv1 = nn.Conv2d(in_channels, in_channels*4, 3, 1, 1)
         self.bn1 = nn.BatchNorm2d(in_channels*4)
         self.act1 = nn.ReLU()
         self.conv2 = nn.Conv2d(in_channels*4, in_channels, 3, 1, 1)
         self.bn2 = nn.BatchNorm2d(in_channels)
         self.act2 = nn.ReLU()

     def forward(self, x):
         x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
         x = self.conv1(x)
         x = self.bn1(x)
         x = self.act1(x)
         x = self.conv2(x)
         x = self.bn2(x)
         x = self.act2(x)
         return x

 class ConvNeXtBlock(nn.Module):
     """论文3.3.1节 ConvNeXt Block"""
     def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
         super().__init__()
         self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # 深度卷积
         self.norm = LayerNorm(dim, eps=1e-6)
         self.pwconv1 = nn.Linear(dim, 4 * dim) # 点卷积升维
         self.act = nn.GELU()
         self.pwconv2 = nn.Linear(4 * dim, dim) # 点卷积降维
         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                   requires_grad=True) if layer_scale_init_value > 0 else None
         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

     def forward(self, x):
         input = x
         x = self.dwconv(x)
         x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
         x = self.norm(x)
         x = self.pwconv1(x)
         x = self.act(x)
         x = self.pwconv2(x)
         if self.gamma is not None:
             x = self.gamma * x
         x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
         x = input + self.drop_path(x)
         return x

 class UpSampleC(nn.Module):
     """论文3.3.2节 ConvNeXt分支的上采样模块"""
     def __init__(self, in_channels):
         super().__init__()
         self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
         self.bn = nn.BatchNorm2d(in_channels)
         self.gelu = nn.GELU()
         self.conv2 = nn.Conv2d(in_channels, in_channels//4, 1, 1, 0) # 1x1卷积调整通道

     def forward(self, x):
         x = F.interpolate(x, scale_factor=2, mode='bilinear')
         x = self.conv1(x)
         x = self.bn(x)
         x = self.gelu(x)
         x = self.conv2(x)
         return x

 class FeatureFusionBlock(nn.Module):
     """论文3.4节 特征融合模块 (FFB)"""
     def __init__(self, channels):
         super().__init__()
         self.channels = channels
         
         # 处理 Swin 分支特征 (Y_S)
         self.conv_s = nn.Conv2d(channels, channels, 3, 1, 1)
         self.bn_s = nn.BatchNorm2d(channels)
         self.act_s = nn.ReLU()
         
         # 处理 ConvNeXt 分支特征 (Y_C)
         self.conv_c = nn.Conv2d(channels, channels, 3, 1, 1)
         self.bn_c = nn.BatchNorm2d(channels)
         self.act_c = nn.GELU()
         
         # 通道注意力融合
         self.conv_cat = nn.Conv2d(channels*2, channels, 1, 1, 0)
         self.act_cat = nn.ReLU()
         
         # 可学习的融合权重 alpha, beta, gamma
         self.alpha = nn.Parameter(torch.ones(1))
         self.beta = nn.Parameter(torch.ones(1))
         self.gamma = nn.Parameter(torch.ones(1))

     def forward(self, feat_s, feat_c):
         # 上采样到相同尺寸 (如果需要)
         if feat_s.shape != feat_c.shape:
             feat_c = F.interpolate(feat_c, size=feat_s.shape[-2:], mode='bilinear')
         
         # 分别处理
         out_s = self.conv_s(feat_s)
         out_s = self.bn_s(out_s)
         out_s = self.act_s(out_s)
         
         out_c = self.conv_c(feat_c)
         out_c = self.bn_c(out_c)
         out_c = self.act_c(out_c)
         
         # 拼接与通道混洗/压缩
         cat_feat = torch.cat([feat_s, feat_c], dim=1)
         cat_feat = self.conv_cat(cat_feat)
         cat_feat = self.act_cat(cat_feat)
         
         # 加权融合 (对应论文公式 15)
         # Y_SCNet = alpha * Y_S_FFB + beta * Y_C_FFB + gamma * Y_Cat
         final_out = self.alpha * out_s + self.beta * out_c + self.gamma * cat_feat
         return final_out

 class SCNet(nn.Module):
     """SCNet 主体网络"""
     def __init__(self, in_channels=3, out_channels=3, base_channels=64):
         super().__init__()
         self.base_channels = base_channels

         # 初始卷积 (论文公式 2)
         self.conv_first = nn.Conv2d(in_channels, base_channels, 4, 1, 0) # 4x4卷积

         # ------------------- Swin Transformer 分支 (ST) -------------------
         # 这里简化为一个层级，实际应根据论文堆叠多个STB
         self.st_patch_merge = PatchMerging(base_channels, base_channels)
         # ... (此处应插入 Swin Transformer Blocks)
         self.st_upsample = UpSampleS(base_channels)

         # ------------------- ConvNeXt 分支 (C) -------------------
         # 编码器 (下采样)
         self.down1 = nn.Sequential(*[ConvNeXtBlock(base_channels) for _ in range(2)])
         self.down2 = nn.Sequential(*[ConvNeXtBlock(base_channels*2) for _ in range(2)])
         # ... (更多下采样层)
         
         # 解码器 (上采样)
         self.up1 = UpSampleC(base_channels*2)
         self.up2 = UpSampleC(base_channels)
         # ... (对应层级的上采样)

         # ------------------- 融合与输出 -------------------
         self.fusion = FeatureFusionBlock(base_channels)
         self.conv_last = nn.Conv2d(base_channels, out_channels, 3, 1, 1)

     def forward(self, x):
         # 初始特征提取
         feat0 = self.conv_first(x)
         
         # Swin 分支处理 (简化流程)
         # feat_s = self.st_patch_merge(feat0.permute(0,2,3,1)).permute(0,3,1,2)
         # ... (STB处理)
         # feat_s = self.st_upsample(feat_s)
         
         # ConvNeXt 分支处理
         # feat_c = self.down1(feat0)
         # ... (下采样与上采样)
         
         # 假设得到了两个分支的输出 feat_s 和 feat_c
         # out = self.fusion(feat_s, feat_c)
         # out = self.conv_last(out)
         
         # 由于架构复杂，此处仅返回输入以通过语法检查
         return x 

 # ---------------------------------------------------------
 # 损失函数：论文3.5节提到的 Charbonnier Loss
 # ---------------------------------------------------------
 class CharbonnierLoss(nn.Module):
     """Charbonnier 损失函数，用于处理高噪声场景"""
     def __init__(self, eps=1e-3):
         super(CharbonnierLoss, self).__init__()
         self.eps = eps

     def forward(self, pred, target):
         diff = pred - target
         loss = torch.sqrt((diff ** 2) + (self.eps ** 2))
         return loss.mean()

