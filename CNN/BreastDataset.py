# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 03:29:27 2024

@author: 27187
"""

import os
import cv2
import torch
from torch.utils.data import Dataset

class BreastDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_dir = os.path.join(root_dir, r"F:\Leslie\uni\Y3\Programming\project\main_dataset\train\images")
        self.density_mask_dir = os.path.join(root_dir, r"F:\Leslie\uni\Y3\Programming\project\main_dataset\train\dense_masks")
        self.breast_mask_dir = os.path.join(root_dir, r"F:\Leslie\uni\Y3\Programming\project\main_dataset\train\breast_masks")
        self.transform = transform

        # 加载文件路径
        self.image_files = os.listdir(self.image_dir)
        self.density_files = self._load_density_csv(self.density_mask_dir)
        self.breast_files = self._load_density_csv(self.breast_mask_dir)

    def _load_density_csv(self, folder_path):
        return {file_name: os.path.join(folder_path, file_name)
                for file_name in os.listdir(folder_path)
                if file_name.endswith(('.png', '.jpg', '.jpeg'))}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        density_path = self.density_files.get(img_name)
        breast_path = self.breast_files.get(img_name)

        # 加载图像数据
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
        density_mask = cv2.imread(density_path, cv2.IMREAD_GRAYSCALE) / 255.0
        breast_mask = cv2.imread(breast_path, cv2.IMREAD_GRAYSCALE) / 255.0

        # 数据增强
        if self.transform:
            img = self.transform(img)
            density_mask = self.transform(density_mask)
            breast_mask = self.transform(breast_mask)

        return (
            torch.tensor(img).unsqueeze(0),  # [C, H, W]
            torch.tensor(density_mask).unsqueeze(0),
            torch.tensor(breast_mask).unsqueeze(0)
        )

train_dataset = BreastDataset(
    root_dir=r"F:\Leslie\uni\Y3\Programming\project\main_dataset\train"
    )