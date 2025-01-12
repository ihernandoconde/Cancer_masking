# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 05:28:08 2025

@author: 27187
"""

# new_lossv0.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):


    def __init__(self, dice_weight=0.5, smooth=1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.smooth = smooth

    @property
    def __name__(self):
       
        return "bcedice_loss"

    def forward(self, pred, target):
        """
        :param pred: [B,1,H,W], not processed by Sigmoid
        :param target: [B,1,H,W], 0/1
        :return: scaler, single task's combined BCE+Dice weighted loss
        """
        bce = F.binary_cross_entropy_with_logits(pred, target)

        pred_sig = torch.sigmoid(pred)
        intersection = (pred_sig * target).sum(dim=[2, 3])
        denominator  = pred_sig.sum(dim=[2, 3]) + target.sum(dim=[2, 3])
        dice_score   = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss    = 1 - dice_score.mean()

        total_loss   = (1 - self.dice_weight)*bce + self.dice_weight*dice_loss
        return total_loss

