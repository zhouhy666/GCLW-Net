import os
import cv2
import numpy as np
from tqdm import tqdm

# =========================
# 路径计算ssdd数据集各缺陷数量占比
# =========================
mask_dir = r"C:\data\SSDD\masks_gray"

# =========================
# 类别设置（按实际修改）
# =========================
num_classes = 5   # 0~4
class_names = {
    0: "背景",
    1: "缺陷1",
    2: "缺陷2",
    3: "缺陷3",
    4: "缺陷4"
}

# =========================
# 统计变量
# =========================
pixel_count = np.zeros(num_classes, dtype=np.int64)
image_count = np.zeros(num_classes, dtype=np.int64)  # 每类出现的图片数

files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg'))]

print(f"总图像数量: {len(files)}")

# =========================
# 遍历
# =========================
for file in tqdm(files):
    path = os.path.join(mask_dir, file)

    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue

    unique_classes = np.unique(mask)

    for cls in unique_classes:
        pixel_count[cls] += np.sum(mask == cls)

        # 统计该类别出现过的图片数量
        image_count[cls] += 1

# =========================
# 像素占比
# =========================
total_pixels = np.sum(pixel_count)

print("\n===== 类别像素占比（含背景） =====")
for cls in range(num_classes):
    ratio = pixel_count[cls] / total_pixels * 100
    print(f"{class_names[cls]}: {ratio:.2f}%")

# =========================
# 去背景占比
# =========================
foreground_pixels = total_pixels - pixel_count[0]

print("\n===== 缺陷像素占比（不含背景） =====")
for cls in range(1, num_classes):
    ratio = pixel_count[cls] / foreground_pixels * 100 if foreground_pixels > 0 else 0
    print(f"{class_names[cls]}: {ratio:.2f}%")

# =========================
# 图片数量统计
# =========================
print("\n===== 各类别出现的图片数量 =====")
for cls in range(1, num_classes):
    print(f"{class_names[cls]}: {image_count[cls]} 张")

# =========================
# 图片占比（更论文友好）
# =========================
total_images = len(files)

print("\n===== 各类别图片占比 =====")
for cls in range(1, num_classes):
    ratio = image_count[cls] / total_images * 100
    print(f"{class_names[cls]}: {ratio:.2f}%")