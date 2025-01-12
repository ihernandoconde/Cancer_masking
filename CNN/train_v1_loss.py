# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 08:12:32 2025

@author: 27187
"""

import os
import torch
import random
import sys

sys.path.append(r"D:\Users\27187\anaconda3\envs\breast_mask\Lib\site-packages\spyder_kernels\customize")

from torch.utils.data import DataLoader
from MammoDataset import MammoDataset  # data load
import Deepdensity.scr.segmentation_models_multi_tasking as smp
from natsort import natsorted
import torch.nn as nn
from sklearn.model_selection import train_test_split

# (1) import self defined BCEDice loss model
from new_lossv0 import BCEDiceLoss

# multitask weighted

def multitask_loss(loss_breast, loss_dense, alpha=0.5):
    return alpha * loss_breast + (1 - alpha) * loss_dense


# (A) training hyperparameter
num_epochs = 30  # iteration time

# image path
root_dir = r"F:\Leslie\uni\Y3\Programming\project\main_dataset\train"
images_path = os.path.join(root_dir, 'images')

all_files = natsorted(os.listdir(images_path))

# 80% train; 20% val
train_files, valid_files = train_test_split(all_files, test_size=0.2, random_state=42)
random.shuffle(train_files)

# load Dataset
train_dataset = MammoDataset(
    root_dir=root_dir,
    files=train_files,
    augmentations=None
)
print("Training dataset size:", len(train_dataset))

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, num_workers=0)

valid_dataset = MammoDataset(
    root_dir=root_dir,
    files=valid_files,
    augmentations=None
)
print("Validation dataset size:", len(valid_dataset))

valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=8, num_workers=0)

# use GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# (B) def model

model = getattr(smp, 'Unet')(
    encoder_name='resnet50',
    encoder_weights='imagenet',
    classes=1,      
    activation=None
)
model = model.to(DEVICE)
model = nn.DataParallel(model)

# (C) loss and optimizer
loss_fn = BCEDiceLoss(dice_weight=0.5)  
optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])
lr_schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)

# (D) metrics
metrics = [
    smp.utils.metrics.Precision(),
    smp.utils.metrics.Recall(),
    smp.utils.metrics.Accuracy(),
    smp.utils.metrics.Fscore(),
    smp.utils.metrics.IoU(threshold=0.5),
]

# (E) def TrainEpoch / ValidEpoch

class TrainEpoch(smp.utils.train.TrainEpoch):
    def __init__(
        self,
        model,
        loss,
        metrics,
        optimizer,
        device='cpu',
        lr_schedular=None,
        verbose=True,
    ):
        # lr_schdular to parent class
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            lr_schedular=lr_schedular,  
            device=device,
            verbose=verbose
        )

    def batch_update(self, x, y, z):
        """
        x: [B,C,H,W]
        y: breast mask  [B,1,H,W]
        z: dense mask   [B,1,H,W]
        return: (loss_breast, loss_dense, weighted_loss, pred_breast, pred_dense)
        """
        self.optimizer.zero_grad()

        prediction_breast, prediction_dense = self.model.forward(x)

        # seperately calc loss for each task
        loss_breast = self.loss(prediction_breast, y)
        loss_dense  = self.loss(prediction_dense, z)

        # multitask loss calc
        weighted_loss = multitask_loss(loss_breast, loss_dense, alpha=0.5)

        weighted_loss.backward()
        self.optimizer.step()

        return loss_breast, loss_dense, weighted_loss, prediction_breast, prediction_dense

    def batch_metrics(self, outputs, y, z):
        """
        outputs: (pred_breast, pred_dense)
        y,z: [B,1,H,W]
        metrics: [Precision, Recall, ...]
        """
        pred_breast, pred_dense = outputs
        return {
            metric.__name__: metric(pred_breast, y)
            for metric in self.metrics
        }

    def epoch_update(self, logs):
        super().epoch_update(logs)  

class ValidEpoch(smp.utils.train.ValidEpoch):
    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(model, loss, metrics, device=device, verbose=verbose)

    def batch_update(self, x, y, z):
        with torch.no_grad():
            prediction_breast, prediction_dense = self.model.forward(x)

            loss_breast = self.loss(prediction_breast, y)
            loss_dense  = self.loss(prediction_dense, z)
            weighted_loss = multitask_loss(loss_breast, loss_dense, alpha=0.5)

        return loss_breast, loss_dense, weighted_loss, prediction_breast, prediction_dense

    def batch_metrics(self, outputs, y, z):
        pred_breast, pred_dense = outputs
        return {
            metric.__name__: metric(pred_breast, y)
            for metric in self.metrics
        }

# (F) main training epoch

train_epoch = TrainEpoch(
    model=model,
    loss=loss_fn,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    lr_schedular=lr_schedular,  
    verbose=True,
)

valid_epoch = ValidEpoch(
    model=model,
    loss=loss_fn,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

os.makedirs("/kaggle/working/results/logs", exist_ok=True)
os.makedirs("/kaggle/working/results/models", exist_ok=True)

train_accuracy = []
valid_accuracy = []
train_loss = []
valid_loss = []

max_score = 0
log_path = "/kaggle/working/results/logs/unet_resnet50_logs.txt"
with open(log_path, 'a+') as logs_file:
    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch + 1}/{num_epochs}")

        # 1) train
        train_logs = train_epoch.run(train_dataloader)
        # 2) val
        valid_logs = valid_epoch.run(valid_dataloader)

        # record loss/acc
        train_loss.append(train_logs['bcedice_loss_weighted'])
        valid_loss.append(valid_logs['bcedice_loss_weighted'])
        train_accuracy.append(train_logs['accuracy'])
        valid_accuracy.append(valid_logs['accuracy'])

        # console print
        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_logs['bcedice_loss_weighted']:.4f}, "
            f"Valid Loss: {valid_logs['bcedice_loss_weighted']:.4f}, "
            f"IoU: {valid_logs['iou_score']:.4f}"
        )

        # file log
        print(
            f"{epoch + 1}\t"
            f"{train_logs['bcedice_loss_breast']:.4f}\t"
            f"{train_logs['bcedice_loss_dense']:.4f}\t"
            f"{train_logs['bcedice_loss_weighted']:.4f}\t"
            f"{train_logs['precision']:.4f}\t{train_logs['recall']:.4f}\t"
            f"{train_logs['accuracy']:.4f}\t{train_logs['fscore']:.4f}\t"
            f"{train_logs['iou_score']:.4f}\t"
            f"{valid_logs['bcedice_loss_breast']:.4f}\t"
            f"{valid_logs['bcedice_loss_dense']:.4f}\t"
            f"{valid_logs['bcedice_loss_weighted']:.4f}\t"
            f"{valid_logs['precision']:.4f}\t{valid_logs['recall']:.4f}\t"
            f"{valid_logs['accuracy']:.4f}\t{valid_logs['fscore']:.4f}\t"
            f"{valid_logs['iou_score']:.4f}",
            file=logs_file
        )

        # save best model
        if valid_logs['iou_score'] > max_score:
            max_score = valid_logs['iou_score']
            torch.save(model, fr"F:\Leslie\uni\Y3\Programming\project\Repo\result\result_epoch{epoch + 1}.pth       ")
            print("Model saved!")

print("Training completed.")