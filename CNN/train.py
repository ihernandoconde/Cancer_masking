# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 17:50:13 2024

@author: 27187
"""

import os
import torch
import random
import sys
sys.path.append(r"D:\Users\27187\anaconda3\envs\breast_mask\Lib\site-packages\spyder_kernels\customize")
from torch.utils.data import DataLoader
from MammoDataset import MammoDataset  # data load
from utils import image_tensor     #tensor from util
import segmentation_models_multi_tasking as smp
from natsort import natsorted
import torch.nn as nn
from sklearn.model_selection import train_test_split


num_epochs = 20  # iteration time

# image path
root_dir = r"F:\Leslie\uni\Y3\Programming\project\main_dataset\train"
images_path = os.path.join(root_dir, 'images')

# dynamic load file name
all_files = natsorted(os.listdir(images_path))

# set train n val： %80 train;20% val
train_files, valid_files = train_test_split(all_files, test_size=0.2, random_state=42)

# random file sequence
random.shuffle(train_files)

# train DS loading
train_dataset = MammoDataset(
    root_dir=root_dir,
    files=train_files,
    augmentations=None
)
print("Training dataset size:", len(train_dataset))

train_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=8, num_workers=0
)

# val DS loading
valid_dataset = MammoDataset(
    root_dir=root_dir,
    files=valid_files,
    augmentations=None
)
print("Validation dataset size:", len(valid_dataset))

valid_dataloader = DataLoader(
    valid_dataset, shuffle=False, batch_size=8, num_workers=0
)

# initialize device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create model
model = getattr(smp, 'Unet')(
    encoder_name='resnet50',
    encoder_weights='imagenet',
    classes=1,
    activation='sigmoid'
)
model = model.to(DEVICE)
model = nn.DataParallel(model)

# define loss function, optimizer n schedular
loss = getattr(smp.utils.losses, 'FocalTverskyLoss')()
optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])
lr_schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)

# def metrices
metrics = [
    smp.utils.metrics.Precision(),
    smp.utils.metrics.Recall(),
    smp.utils.metrics.Accuracy(),
    smp.utils.metrics.Fscore(),
    smp.utils.metrics.IoU(threshold=0.5),
]

# def train n val iteration
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    lr_schedular=lr_schedular,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

# create log n model directory
os.makedirs("/kaggle/working/results/logs", exist_ok=True)
os.makedirs("/kaggle/working/results/models", exist_ok=True)

# ini train
train_accuracy = []
valid_accuracy = []
train_loss = []
valid_loss = []

# train
max_score = 0
with open("/kaggle/working/results/logs/unet_resnet50_logs.txt", 'a+') as logs_file:
    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch + 1}/{num_epochs}")
        
        # train
        train_logs = train_epoch.run(train_dataloader)
        
        # val
        valid_logs = valid_epoch.run(valid_dataloader)

        # loss function
        train_accuracy.append(train_logs['accuracy'])
        valid_accuracy.append(valid_logs['accuracy'])
        train_loss.append(train_logs['focal_tversky_loss_weighted'])
        valid_loss.append(valid_logs['focal_tversky_loss_weighted'])

        # print log
        print(
            f"Epoch {epoch + 1} | Train Loss: {train_logs['focal_tversky_loss_weighted']:.4f}, "
            f"Valid Loss: {valid_logs['focal_tversky_loss_weighted']:.4f}, "
            f"IoU: {valid_logs['iou_score']:.4f}"
        )
        print(
            f"{epoch + 1}\t{train_logs['focal_tversky_loss_breast']:.4f}\t"
            f"{train_logs['focal_tversky_loss_dense']:.4f}\t"
            f"{train_logs['focal_tversky_loss_weighted']:.4f}\t"
            f"{train_logs['precision']:.4f}\t{train_logs['recall']:.4f}\t"
            f"{train_logs['accuracy']:.4f}\t{train_logs['fscore']:.4f}\t"
            f"{train_logs['iou_score']:.4f}\t"
            f"{valid_logs['focal_tversky_loss_breast']:.4f}\t"
            f"{valid_logs['focal_tversky_loss_dense']:.4f}\t"
            f"{valid_logs['focal_tversky_loss_weighted']:.4f}\t"
            f"{valid_logs['precision']:.4f}\t{valid_logs['recall']:.4f}\t"
            f"{valid_logs['accuracy']:.4f}\t{valid_logs['fscore']:.4f}\t"
            f"{valid_logs['iou_score']:.4f}",
            file=logs_file
        )

        # save best mod
        if valid_logs['iou_score'] > max_score:
            max_score = valid_logs['iou_score']
            torch.save(model, r"F:\Leslie\uni\Y3\Programming\project\CNN\result_epoch{epoch + 1}.pth")
            print("Model saved!")

print("Training completed.")
