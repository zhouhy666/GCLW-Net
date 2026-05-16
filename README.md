# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

We propose a global context and boundary-aware semantic segmentation framework tailored for industrial steel surfaces. This method integrates a Global Context Atrous Spatial Pyramid Pooling module for multi-scale feature aggregation, a lightweight adaptive attention mechanism for feature enhancement, and a boundary-constrained Dice loss function to optimize defect contours. Experiments on the NEU-SEG and SSDD datasets demonstrate that the model achieves an mIoU of 81.5% and an mAP of 91.3%, outperforming mainstream segmentation networks. The proposed approach significantly improves the localization accuracy and boundary segmentation performance for low-contrast and multi-scale defects in practical industrial scenarios.

## Core Commands

### Training the Model
```bash
# Single model training
python train.py --dataset pascal --model_type UNet --batch_size 8 --epochs 100

# Specify GPUs
python train.py --dataset pascal --model_type UNet --gpu-ids 0,1

# Resume training from checkpoint
python train.py --dataset pascal --model_type UNet --resume /path/to/checkpoint.pth.tar

# Validate the model
python val.py --model_type UNet --resume /path/to/model_best.pth.tar

# Inference / Prediction
python predict.py --model_type UNet --resume /path/to/model_best.pth.tar

##Environment Verification
```bash
python env_test.py
```

## Architecture Overview
### Directory Structure
```
GCLW-UNet/
├── blocks/              # Pluggable modules (Attention, Convolution improvements, etc.)
├── models/              # Model definitions
│   ├── unet_model.py    # Main model file (contains all improved models)
│   ├── unet_parts.py    # Basic components (DoubleConv, Down, Up, OutConv)
│   └── model_zoo.py     # Model registry
├── dataloaders/         # Data loading
├── utils/               # Utilities (loss, metrics, saver)
├── NEU_Seg_data/        # NEU-SEG dataset
├── train.py/val.py/predict.py  # Training/Validation/Inference scripts
└── ssddval.py           # Validation script for SSDD dataset
```

### Module Call Chain
1.blocks/: Contains 30+ independent modules (CBAM, SimAM, RepLKBlock, FastKAN, etc.), dynamically imported via blocks/__init__.py.
2.unet/unet_model.py: Defines all UNet variant models, directly using modules via from blocks import *.
3.unet/model_zoo.py: Unified registry MODEL_ZOO, retrieves model classes by string name.
4.train.py: Uses the --model_type argument to fetch the model class from globals() or MODEL_ZOO.

### Model Registration Process
Three-step workflow to add a new model:
1.Create a new module under blocks/ (optional).
2.Define the model class in unet/unet_model.py.
3.Register it in the MODEL_ZOO dictionary within unet/model_zoo.py.

### Data Flow
```
mypath.py (Path configuration)
  → dataloaders/__init__.py (make_data_loader factory function)
  → datasets/mydataset.py (Custom Dataset class)
```

### Training Pipeline
```
train.py → Trainer class
  → training(epoch): Mixed precision training (AMP) + tqdm progress bar
  → validation(epoch): Evaluator calculates mIoU/mPA/Dice
  → Saver: Saves checkpoints and TensorBoard logs
```

## Key Configurations

### Dataset Configuration
-NEU_Seg_data/ (NEU-SEG)
-SSDD_data/ (SSDD)

### Loss Functions
-CrossEntropy (Default)
-Focal Loss
-Dice Loss

### Learning Rate Scheduler
- poly (Default)
- step
- cos (Cosine annealing)

## Supported Models
- Check the MODEL_ZOO dictionary in unet/model_zoo.py, which includes:
- UNet baseline and variants
- Comparison models (UNetPP, DeepLabV3Plus, SegNet, PSPNet, BiSeNet, OCNet, etc.)
