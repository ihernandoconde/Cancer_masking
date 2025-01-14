# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 00:19:21 2025

@author: 27187
"""

import torch
import sys

sys.path.append(str(r'F:\Leslie\uni\Y3\Programming\project\Repo\cancer_masking\CNN\newest_version'))

def test_bcedice_loss(BCEDiceLoss):
    """
    A basic unit test function for BCEDiceLoss.
    It tests:
    1) Normal usage with correct shapes
    2) Mismatched shapes
    3) Out-of-range target values
    4) Single channel constraint
    """

    loss_fn = BCEDiceLoss(dice_weight=0.5, smooth=1e-5)

    print("=== Test 1: Normal usage with correct shapes ===")
    try:
        # Create a normal input
        pred   = torch.randn(2, 1, 256, 256)  # [B=2, C=1, H=256, W=256]
        target = torch.zeros_like(pred)        # target in [0,1], here all 0
        val = loss_fn(pred, target)
        assert val.dim() == 0, "Loss should be a scalar (0-dim tensor)."
        print("Test 1 passed: normal usage returns a scalar loss.")
    except Exception as e:
        print(f"Test 1 failed: {e}")

    print("=== Test 2: Mismatched shapes ===")
    try:
        pred   = torch.randn(2, 1, 256, 256)
        target = torch.zeros(2, 1, 128, 128)   # mismatch
        _ = loss_fn(pred, target)
        print("Test 2 failed: expected an error but none was raised.")
    except ValueError as ve:
        # ValueError raised about shape mismatch
        print(f"Test 2 passed: caught expected ValueError => {ve}")
    except Exception as e:
        print(f"Test 2 failed: caught unexpected error => {e}")

    print("=== Test 3: Out-of-range target values ===")
    try:
        pred   = torch.randn(2, 1, 256, 256)
        target = torch.full((2,1,256,256), 2.0)  # target=2.0, out of [0,1]
        _ = loss_fn(pred, target)
        print("Test 3 failed: expected an error but none was raised.")
    except ValueError as ve:
        # ValueError raised about target out of [0,1]
        print(f"Test 3 passed: caught expected ValueError => {ve}")
    except Exception as e:
        print(f"Test 3 failed: caught unexpected error => {e}")

    print("=== Test 4: Single channel constraint ===")
    try:
        # Make pred with shape [B, C=2, H, W] which didn't match with expected single channel
        pred   = torch.randn(2, 2, 256, 256)
        target = torch.zeros(2, 2, 256, 256)
        _ = loss_fn(pred, target)
        print("Test 4 failed: expected an error but none was raised.")
    except ValueError as ve:
        # ValueError raised about single-channel usage
        print(f"Test 4 passed: caught expected ValueError => {ve}")
    except Exception as e:
        print(f"Test 4 failed: caught unexpected error => {e}")

    print("\nAll tests completed.\n")

if __name__ == "__main__":

    from loss_v1 import BCEDiceLoss  

    test_bcedice_loss(BCEDiceLoss)
