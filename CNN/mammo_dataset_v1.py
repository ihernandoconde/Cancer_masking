# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 19:54:34 2025

@author: 27187
"""

import pandas as pd
import os
import sys
sys.path.append(r"F:\Leslie\uni\Y3\Programming\project\CNN\Sample")
from natsort import natsorted
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A

"""
 this code is the code that preprocessed data, including format converting, data enhancement, and data reading.
 This class will be called during training.
 
"""

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
        """
        :param root_dir: root path，including images/breast_masks/dense_masks 
        """

        # common error check
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"root_dir does not exist: {root_dir}")

        self.augmentations = augmentations

        # img、mask、contour path
        try:
            self.images = natsorted([
                os.path.join(root_dir, 'images', f) 
                for f in (files or os.listdir(os.path.join(root_dir, 'images')))
            ])
            self.masks = natsorted([
                os.path.join(root_dir, 'breast_masks', f) 
                for f in (files or os.listdir(os.path.join(root_dir, 'breast_masks')))
            ])
            self.contours = natsorted([
                os.path.join(root_dir, 'dense_masks', f) 
                for f in (files or os.listdir(os.path.join(root_dir, 'dense_masks')))
            ])
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Failed to list files. Check subfolders exist under {root_dir}: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error reading file lists: {e}")

        # check if the nums of img match
        if not (len(self.images) == len(self.masks) == len(self.contours)):
            raise ValueError(
                f"Number of images, masks, contours not match! "
                f"images={len(self.images)}, masks={len(self.masks)}, contours={len(self.contours)}"
            )
        if len(self.images) == 0:
            raise ValueError("No files found to load in the dataset.")

        # def defalut transform method
        self.to_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # def data enhancement method
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
        """
        return: (image_tensor, breast_mask_tensor, dense_mask_tensor)
        """

        # path
        image_path = self.images[index]
        mask_path  = self.masks[index]
        contour_path = self.contours[index]

        # debugging info (could be commented)
        #print(f"Loading image: {image_path}")

        # load img (3 channel), mask(gery scale), contour(gery scale)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to load breast mask: {mask_path}")

        contour = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
        if contour is None:
            raise FileNotFoundError(f"Failed to load dense mask: {contour_path}")

        # convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # use albumentations enhancement
        if self.augmentations:
            
            try:
                augmented = self.aug_pipeline(
                    image=image,
                    masks=[mask, contour]
                )
                image_aug = augmented['image']
                mask_aug, contour_aug = augmented['masks']
            except Exception as e:
                raise RuntimeError(f"Augmentation error with albumentations: {e}")

            image   = image_aug
            mask    = mask_aug
            contour = contour_aug

        # convert to torch.tensor
        #    cv2 to PIL + resize(256,256) + to_tensor
        try:
            image_tensor   = self.to_tensor(image)
            breast_tensor  = self.to_tensor(mask)
            dense_tensor   = self.to_tensor(contour)
        except Exception as e:
            raise RuntimeError(f"Error in transform to_tensor: {e}")

        # return
        return image_tensor, breast_tensor, dense_tensor
