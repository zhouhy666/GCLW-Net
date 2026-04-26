import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import torch

class MyDataset(Dataset):
    """
    SSDD 自定义语义分割数据集
    VOC风格目录：
    - JPEGImages/
    - SegmentationClass/
    - ImageSets/Segmentation/train.txt / val.txt / test.txt
    """
    NUM_CLASSES = 5  # 背景 + 4 类缺陷

    def __init__(self, root, split="train", transform=True):
        self.root = root
        self.split = split
        self.transform = transform

        self.image_dir = os.path.join(root, 'JPEGImages')
        self.mask_dir = os.path.join(root, 'SegmentationClass')
        split_f = os.path.join(root, 'ImageSets', 'Segmentation', f'{split}.txt')
        if not os.path.exists(split_f):
            raise ValueError(f"Split 文件不存在: {split_f}")

        with open(split_f, 'r') as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(self.image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(self.mask_dir, x + ".png") for x in file_names]

        assert len(self.images) == len(self.masks)

        if self.transform:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
            ])
        else:
            self.transforms = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = np.array(Image.open(self.masks[index])).astype(np.uint8)

        if self.transforms:
            img = self.transforms(img)

        # 注意这里 key 要和训练代码一致
        return {'image': img, 'label': torch.from_numpy(mask).long()}

    # --------------------------
    # 辅助方法：统计 mask 类别
    # --------------------------
    def check_classes(self):
        all_classes = set()
        for mask_path in self.masks:
            mask = np.array(Image.open(mask_path))
            classes = np.unique(mask)
            all_classes.update(classes.tolist())
        print(f"[{self.split}] 数据集中共有类别: {sorted(all_classes)}")
        return sorted(all_classes)

    # --------------------------
    # 辅助方法：可视化 mask（彩色叠加）
    # --------------------------

