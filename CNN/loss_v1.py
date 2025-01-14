# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 19:40:28 2025

@author: 27187
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    """
    single channel output: [B,1,H,W] output, use BCE+Dice combined loss function to optimize model's performance
    """

    def __init__(self, dice_weight=0.5, smooth=1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.smooth = smooth

    @property
    def __name__(self):
        return "bcedice_loss"

    def forward(self, pred, target):
        """
        :param pred:   [B,1,H,W]，net output（logits），didn't pass sigmoid
        :param target: [B,1,H,W]， 0/1
        :return: scalar，total loss = BCEWithLogitsLoss + DiceLoss weighted result
        """
        # basic error handling: dimension & shape
        if pred.ndim != 4 or target.ndim != 4:
            raise ValueError(
                f"BCEDiceLoss expects 4D tensors (B,C,H,W). Got pred {pred.shape}, target {target.shape}."
            )
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
            )
        if pred.shape[1] != 1 or target.shape[1] != 1:
            raise ValueError(
                "BCEDiceLoss is for single-channel seg only. "
                f"Got pred.shape[1]={pred.shape[1]}, target.shape[1]={target.shape[1]}."
            )

        # check target value range: target ∈ [0,1].
        min_val, max_val = target.min().item(), target.max().item()
        if min_val < 0 or max_val > 1:
            raise ValueError(
                f"Target values out of [0,1] range. min={min_val}, max={max_val}"
            )

        # use try/except calc BCE
        try:
            bce = F.binary_cross_entropy_with_logits(pred, target)
        except RuntimeError as e:
            raise RuntimeError(
                f"Error in BCE calculation. pred={pred.shape}, target={target.shape}: {e}"
            )

        # use try/except calc Dice
        try:
            pred_sig = torch.sigmoid(pred)
            intersection = (pred_sig * target).sum(dim=[2, 3])
            denominator  = pred_sig.sum(dim=[2, 3]) + target.sum(dim=[2, 3])
            dice_score   = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
            dice_loss    = 1 - dice_score.mean()
        except RuntimeError as e:
            raise RuntimeError(
                f"Error in Dice calculation. pred_sig={pred_sig.shape}, target={target.shape}: {e}"
            )

        # weighted sum
        total_loss = (1 - self.dice_weight) * bce + self.dice_weight * dice_loss

        return total_loss
