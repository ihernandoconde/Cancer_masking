# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 15:46:03 2024

@author: 27187
"""

import pandas as pd
import os
import sys
sys.path.append(r"F:\Leslie\uni\Y3\Programming\project\CNN\Sample")
from natsort import natsorted
import glob
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_multi_tasking as smp
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from natsort import natsorted

# Root path of files
root_dir = r"F:\Leslie\uni\Y3\Programming\project\main_dataset\train"
images_path = os.path.join(root_dir, 'images')
dense_masks_path = os.path.join(root_dir, 'dense_masks')
breast_masks_path = os.path.join(root_dir, 'breast_masks')

# Dynamic loading of file names
image_files = natsorted(os.listdir(images_path))
dense_mask_files = natsorted(os.listdir(dense_masks_path))
breast_mask_files = natsorted(os.listdir(breast_masks_path))

# Check if file names match
assert image_files == dense_mask_files == breast_mask_files, "File names do not match!"

# Create a DataFrame to simulate train.csv structure
train_df = pd.DataFrame({
    'Filename': image_files,
    'ImagePath': [os.path.join(images_path, f) for f in image_files],
    'DenseMaskPath': [os.path.join(dense_masks_path, f) for f in dense_mask_files],
    'BreastMaskPath': [os.path.join(breast_masks_path, f) for f in breast_mask_files],
    'Density': [0.0] * len(image_files)  # Placeholder for density values
})

# Sort by Density column (if relevant)
train_df.sort_values(by='Density', inplace=True)

# Set Filename as index
train_df.set_index('Filename', inplace=True)

# Add prediction columns (optional, for training purposes)
train_df['pred1'] = 0
train_df['pred2'] = 0

# Display first few rows for debugging
print("Train DataFrame:")
print(train_df.head())

class MammoDataset(Dataset):
    def __init__(self, root_dir, files=None, augmentations=False):

        self.augmentations = augmentations

        # 加载图像、乳腺掩膜和密度掩膜路径
        self.images = natsorted([os.path.join(root_dir, 'images', f) for f in (files or os.listdir(os.path.join(root_dir, 'images')))])
        self.masks = natsorted([os.path.join(root_dir, 'breast_masks', f) for f in (files or os.listdir(os.path.join(root_dir, 'breast_masks')))])
        self.contours = natsorted([os.path.join(root_dir, 'dense_masks', f) for f in (files or os.listdir(os.path.join(root_dir, 'dense_masks')))])

        # 检查路径是否一致
        assert len(self.images) == len(self.masks) == len(self.contours), "图像和掩膜数量不匹配！"

        # 定义默认的转换方式
        self.to_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # 定义增强方式
        self.aug_pipeline = A.Compose([
            A.ShiftScaleRotate(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.5),
            A.GaussianBlur(p=0.5),
            A.ElasticTransform(),
        ], p=0.8)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
    # 加载图像路径
        image_path = self.images[index]
        print(f"Loading image: {image_path}")  # 调试信息
        image = cv2.imread(image_path, 1)
    
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[index], 0)
        contour = cv2.imread(self.contours[index], 0)
    
        return self.to_tensor(image), self.to_tensor(mask), self.to_tensor(contour)