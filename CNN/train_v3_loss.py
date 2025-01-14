# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 03:07:50 2025

@author: 27187
"""

import os
import torch
import random
import sys

# (A) import the env and required dependencies
sys.path.append(r"D:\Users\27187\anaconda3\envs\breast_mask\Lib\site-packages\spyder_kernels\customize")
sys.path.append(r"F:\Leslie\uni\Y3\Programming\project\Repo\cancer_masking\CNN")

from torch.utils.data import DataLoader
from mammo_dataset_v1 import MammoDataset  # data load
import Deepdensity.scr.segmentation_models_multi_tasking as smp
from natsort import natsorted
import torch.nn as nn
from sklearn.model_selection import train_test_split

"""
 this code is the main code for training CNN, including define CNN structure and hyperparameters, training and validating CNN, 
 and log the training epoch and best result
 the CNN used ResNet50 as encoders, use multitask structyure to achieve to task segmentation, 
 and use BCE+Disce loss function to compute.
 
"""

# import self-defined BCEDice loss function
from loss_v1 import BCEDiceLoss

# error handling: check existence of folder/path
def check_dataset_folder(folder_path):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Dataset folder not found: {folder_path}")
    files = os.listdir(folder_path)
    if len(files) == 0:
        raise ValueError(f"No files found in the dataset folder: {folder_path}")

def multitask_loss(loss_breast, loss_dense, alpha=0.5):
    return alpha * loss_breast + (1 - alpha) * loss_dense

# training hyperparameters
num_epochs = 20  # iteration time

# image path
root_dir = r"F:\Leslie\uni\Y3\Programming\project\main_dataset\train"
images_path = os.path.join(root_dir, 'images')

# Error handling: check dataset folder
check_dataset_folder(root_dir)
check_dataset_folder(images_path)

all_files = natsorted(os.listdir(images_path))

# 80% train; 20% val
train_files, valid_files = train_test_split(all_files, test_size=0.2, random_state=42)
random.shuffle(train_files)

# error handling: check splitted files
if len(train_files) == 0:
    raise ValueError("No training files found after splitting! Check dataset.")
if len(valid_files) == 0:
    raise ValueError("No validation files found after splitting! Check dataset.")

# load sele-defined dataset load function
train_dataset = MammoDataset(
    root_dir=root_dir,
    files=train_files,
    augmentations=None
)
print("Training dataset size:", len(train_dataset))

# check if train dataset exist
if len(train_dataset) == 0:
    raise ValueError("Training dataset is empty!")

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, num_workers=0)

valid_dataset = MammoDataset(
    root_dir=root_dir,
    files=valid_files,
    augmentations=None
)
print("Validation dataset size:", len(valid_dataset))

#check if val dataset exist
if len(valid_dataset) == 0:
    raise ValueError("Validation dataset is empty!")

valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=8, num_workers=0)

# use GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Warning: CUDA not available, using CPU might be slower.")

# (B) define model
model = getattr(smp, 'Unet')(
    encoder_name='resnet50',# encoder use resnet50 import from DeepDensity package
    encoder_weights='imagenet', # initial weights use imagenet
    classes=1,
    activation='sigmoid'
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

# (E) define TrainEpoch / ValidEpoch
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
        Updates model parameters for a single batch during training.
        - x: input images
        - y: ground truth for breast region segmentation
        - z: ground truth for dense tissue segmentation
        """
        self.optimizer.zero_grad()# Reset gradients
        prediction_breast, prediction_dense = self.model.forward(x)# Forward pass
        
        # Compute multitask losses
        loss_breast = self.loss(prediction_breast, y)
        loss_dense  = self.loss(prediction_dense, z)
        weighted_loss = multitask_loss(loss_breast, loss_dense, alpha=0.5)
        
        # Backpropagation and optimizer step
        weighted_loss.backward()
        self.optimizer.step()

        return loss_breast, loss_dense, weighted_loss, prediction_breast, prediction_dense

    def batch_metrics(self, outputs, y, z):
        """
        Compute metrics for a batch.
        - outputs: predictions (breast and dense tissue)
        - y: ground truth for breast region
        - z: ground truth for dense tissue
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
        """
        Updates model parameters for a single batch during validation.
        """
        with torch.no_grad(): # Disable gradient computation
            prediction_breast, prediction_dense = self.model.forward(x)
            
            # Compute multitask losses
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

# (F) main training
# Initializing custom training and validation classes.
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

# save log & model
local_logs_dir = r"F:\Leslie\uni\Y3\Programming\project\Repo\train_logs"
local_models_dir = r"F:\Leslie\uni\Y3\Programming\project\Repo\train_models"
os.makedirs(local_logs_dir, exist_ok=True)
os.makedirs(local_models_dir, exist_ok=True)

# Initializing containers for tracking loss and accuracy
train_accuracy = []
valid_accuracy = []
train_loss = []
valid_loss = []

# Start training loop
max_score = 0
log_path = os.path.join(local_logs_dir, "unet_resnet50_logs.txt")

try:
    with open(log_path, 'a+') as logs_file:
        for epoch in range(num_epochs):
            print(f"\nEpoch: {epoch + 1}/{num_epochs}")

            # 1) train
            train_logs = train_epoch.run(train_dataloader)
            # 2) val
            valid_logs = valid_epoch.run(valid_dataloader)

            # log losses and accuracy
            train_loss.append(train_logs['bcedice_loss_weighted'])
            valid_loss.append(valid_logs['bcedice_loss_weighted'])
            train_accuracy.append(train_logs['accuracy'])
            valid_accuracy.append(valid_logs['accuracy'])

            # print epoch summary
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
                model_path = os.path.join(local_models_dir, f"result_epoch{epoch + 1}.pth")
                torch.save(model, model_path)
                print("Model saved!")

    print("Training completed.")

except KeyboardInterrupt:
    print("\nUser interrupted training manually. Exiting gracefully...")
except RuntimeError as e:
    if 'CUDA out of memory' in str(e):
        print("Error: CUDA out of memory! Try reducing batch size or image size.")
    else:
        print(f"RuntimeError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
