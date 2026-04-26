import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from dataloaders.datasets.mydataset import MyDataset
from torch.utils.data import DataLoader
from unet.deeplabv3 import DeepLabV3Plus
from unet.unet_model import *

# ===============================
# 参数
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'run/mydataset/UNetASPP_CBAM_0320_1621/model_best.pth.tar'
DATA_ROOT = r"C:\shiyan\UNet-NEU-SEG-main\ssdd"
NUM_CLASSES = 5
IOU_THRESHOLD = 0.5  # AP50计算的IoU阈值

# ===============================
# 加载模型
# ===============================
model = UNetASPP_CBAM(n_channels=3, n_classes=NUM_CLASSES)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

# 适配多卡训练的权重（去除module.前缀）
state_dict = checkpoint['state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)
model.to(DEVICE)
model.eval()

print("模型加载完成！")

# ===============================
# 数据加载
# ===============================
val_dataset = MyDataset(root=DATA_ROOT, split='val', transform=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)


# ===============================
# 评价指标函数（最终修复版）
# ===============================
def fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def compute_iou_per_sample(gt, pred, n_class):
    """计算单张图像每个类别的IoU（样本级IoU）"""
    iou_list = []
    gt = np.array(gt).astype(np.uint8)
    pred = np.array(pred).astype(np.uint8)

    for cls in range(n_class):
        gt_cls = (gt == cls).astype(np.float32)
        pred_cls = (pred == cls).astype(np.float32)

        if np.sum(gt_cls) == 0 and np.sum(pred_cls) == 0:
            iou = 1.0  # 无该类别，IoU记为1.0
        elif np.sum(gt_cls) == 0 or np.sum(pred_cls) == 0:
            iou = 0.0  # 有预测/真实但无交集，IoU记为0.0
        else:
            intersection = np.sum(gt_cls * pred_cls)
            union = np.sum(gt_cls) + np.sum(pred_cls) - intersection
            iou = intersection / union if union > 0 else 0.0

        iou_list.append(float(iou))

    return np.array(iou_list, dtype=np.float32)


def compute_ap50_per_class(iou_list_per_cls):
    """计算每个类别的AP50（鲁棒性优化）"""
    # 过滤无效值
    iou_list_per_cls = [float(iou) for iou in iou_list_per_cls if not np.isnan(iou) and not np.isinf(iou)]
    if len(iou_list_per_cls) == 0:
        return 0.0

    # 降序排序
    sorted_iou = np.array(iou_list_per_cls, dtype=np.float32)
    sorted_iou = np.sort(sorted_iou)[::-1]

    # 计算召回率和精度
    recall = np.arange(1, len(sorted_iou) + 1) / len(sorted_iou)
    is_above_thresh = sorted_iou >= IOU_THRESHOLD
    precision = np.cumsum(is_above_thresh) / np.arange(1, len(sorted_iou) + 1)

    # 精度插值
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # 计算AP
    if len(recall) == 1:
        ap = precision[0] * recall[0]
    else:
        ap = np.sum((recall[1:] - recall[:-1]) * precision[:-1])

    return float(ap)


def compute_iou_per_class(hist):
    """计算每个类别的整体IoU"""
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    iu = np.nan_to_num(iu)
    return iu


def compute_dice_per_class(hist):
    """计算每个类别的Dice系数"""
    dice = 2 * np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0))
    dice = np.nan_to_num(dice)
    return dice


def compute_metrics(hist, all_iou_per_sample, val_dataset):
    """整合所有评估指标（修复i未定义问题）"""
    # 基础指标
    acc = np.diag(hist).sum() / hist.sum() if hist.sum() > 0 else 0.0
    acc_cls = np.nanmean(np.diag(hist) / hist.sum(axis=1)) if hist.sum(axis=1).sum() > 0 else 0.0
    iu = compute_iou_per_class(hist)
    miou = np.nanmean(iu)
    fw_iou = (hist.sum(axis=1) / hist.sum() * iu).sum() if hist.sum() > 0 else 0.0

    # mDice
    dice = compute_dice_per_class(hist)
    mdice = np.nanmean(dice)

    # mAP50（核心修复：正确遍历样本索引）
    ap50_list = []
    for cls in range(NUM_CLASSES):
        cls_iou_list = []
        # 遍历每个样本，同时获取索引和IoU
        for idx, sample_iou in enumerate(all_iou_per_sample):
            iou_val = float(sample_iou[cls]) if isinstance(sample_iou, (np.ndarray, list)) else 0.0
            # 检查该样本是否真的包含当前类别（修复i未定义问题）
            sample_label = val_dataset[idx]['label'].numpy() if idx < len(val_dataset) else np.array([])
            has_cls = np.sum(sample_label == cls) > 0
            # 仅保留：有该类别标注 或 IoU≠1.0（避免过滤有效样本）
            if not (iou_val == 1.0 and not has_cls):
                cls_iou_list.append(iou_val)

        # 计算该类别AP50
        ap50 = compute_ap50_per_class(cls_iou_list)
        ap50_list.append(ap50)

    map50 = np.nanmean(ap50_list) if len(ap50_list) > 0 else 0.0

    return acc, acc_cls, miou, fw_iou, map50, mdice


# ===============================
# 验证
# ===============================
hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
all_iou_per_sample = []

with torch.no_grad():
    for sample in tqdm(val_loader):
        images = sample['image'].to(DEVICE)
        targets = sample['label'].to(DEVICE)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()

        for lt, lp in zip(targets, preds):
            hist += fast_hist(lt.flatten(), lp.flatten(), NUM_CLASSES)
            sample_iou = compute_iou_per_sample(lt, lp, NUM_CLASSES)
            all_iou_per_sample.append(sample_iou)

# ===============================
# 输出结果
# ===============================
# 关键：传入val_dataset到compute_metrics
acc, acc_cls, miou, fw_iou, map50, mdice = compute_metrics(hist, all_iou_per_sample, val_dataset)

print("\n====== 验证结果 ======")
print(f"Pixel Acc: {acc:.4f}")
print(f"Class Acc: {acc_cls:.4f}")
print(f"mIoU: {miou:.4f}")
print(f"FWIoU: {fw_iou:.4f}")
print(f"mAP50%: {map50:.4f}")
print(f"mDice: {mdice:.4f}")

# 打印每个类别的AP50（修复i未定义）
print("\n====== 每个类别的AP50 ======")
for cls in range(NUM_CLASSES):
    cls_iou_list = []
    for idx, sample_iou in enumerate(all_iou_per_sample):
        iou_val = float(sample_iou[cls]) if isinstance(sample_iou, (np.ndarray, list)) else 0.0
        sample_label = val_dataset[idx]['label'].numpy() if idx < len(val_dataset) else np.array([])
        has_cls = np.sum(sample_label == cls) > 0
        if not (iou_val == 1.0 and not has_cls):
            cls_iou_list.append(iou_val)
    cls_ap50 = compute_ap50_per_class(cls_iou_list)
    print(f"类别 {cls} AP50: {cls_ap50:.4f}")