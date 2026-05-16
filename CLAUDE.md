# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

NEU-SEG 是一个基于 UNet 的钢材表面缺陷语义分割项目，支持多种改进模型。主要使用 NEU-SEG 数据集（4 类：背景、夹渣、斑块、划痕）。

## 核心命令

### 训练模型
```bash
# 单模型训练
python train.py --dataset pascal --model_type UNet --batch_size 8 --epochs 75

# 指定 GPU
python train.py --dataset pascal --model_type UNet --gpu-ids 0,1

# 断点续训
python train.py --dataset pascal --model_type UNet --resume /path/to/checkpoint.pth.tar

# 验证模型
python val.py --model_type UNet --resume /path/to/model_best.pth.tar

# 推理预测
python predict.py --model_type UNet --resume /path/to/model_best.pth.tar
```

### 环境验证
```bash
python env_test.py
```

## 架构概览

### 目录结构
```
UNet-NEU-SEG-main/
├── blocks/              # 可插拔模块（注意力、卷积改进等）
├── unet/                # 模型定义
│   ├── unet_model.py    # 主模型文件（含所有改进 UNet 变种）
│   ├── unet_parts.py    # 基础组件（DoubleConv, Down, Up, OutConv）
│   └── model_zoo.py     # 模型注册表
├── dataloaders/         # 数据加载
├── utils/               # 工具（loss, metrics, saver）
├── data_wenjian/        # NEU-SEG 数据集
├── train.py/val.py/predict.py  # 训练/验证/推理脚本
└── mypath.py            # 数据集路径配置
```

### 模块调用链
1. **blocks/**: 包含 30+ 个独立模块（CBAM, SimAM, RepLKBlock, FastKAN 等），通过 `blocks/__init__.py` 动态导入
2. **unet/unet_model.py**: 定义所有 UNet 变种模型，从 `blocks import *` 直接使用模块
3. **unet/model_zoo.py**: 统一注册表 `MODEL_ZOO`，通过字符串名称获取模型类
4. **train.py**: 使用 `--model_type` 参数从 `globals()` 或 `MODEL_ZOO` 获取模型类

### 模型注册流程
添加新模型的三步流程：
1. 在 `blocks/` 下创建新模块（可选）
2. 在 `unet/unet_model.py` 中定义模型类
3. 在 `unet/model_zoo.py` 中注册到 `MODEL_ZOO` 字典

### 数据流
```
mypath.py (路径配置)
  → dataloaders/__init__.py (make_data_loader 工厂函数)
  → datasets/mydataset.py (自定义 Dataset 类)
```

### 训练流程
```
train.py → Trainer 类
  → training(epoch): 混合精度训练 (AMP) + tqdm 进度条
  → validation(epoch): Evaluator 计算 mIoU/mPA/Dice
  → Saver: 保存 checkpoint 和 TensorBoard 日志
```

## 关键配置

### 数据集配置 (mypath.py)
- `pascal` → `data_wenjian/` (NEU-SEG)
- `mydataset` → `data_magnetic/` (MagneticTile)

### 损失函数
- CrossEntropy (默认)
- Focal Loss
- Dice Loss

### 学习率调度器
- poly (默认)
- step
- cos (余弦退火)

## 已支持模型
查看 `unet/model_zoo.py` 的 `MODEL_ZOO` 字典，包含：
- UNet 基线及变种（UNetCBAM, UNetSimAM, UNet_RepLKBlock_FastKAN 等）
- 对比模型（UNetPP, DeepLabV3Plus, SegNet, PSPNet, BiSeNet, OCNet等）
