# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 00:01:44 2025

@author: 27187
"""

import pytest
import numpy as np
import cv2
import sys

from pathlib import Path

sys.path.append(str(r'F:\Leslie\uni\Y3\Programming\project\Repo\cancer_masking\CNN\newest_version'))
from mammo_dataset_v1 import MammoDataset  

@pytest.fixture
def fake_dataset_root(tmp_path):
    """
    In pytest's temp folder，create images/breast_masks/dense_masks，
    and create several test img to test mammo_dataset_v1
    return: root_dir (str)
    """
    # create sub folders
    images_dir = tmp_path / "images"
    breast_dir = tmp_path / "breast_masks"
    dense_dir  = tmp_path / "dense_masks"

    images_dir.mkdir()
    breast_dir.mkdir()
    dense_dir.mkdir()

    # create test img
    num_files = 3
    for i in range(num_files):
        filename = f"test_{i}.png"

        # test img：pure color/different grey scale b using numpy and cv2
       
        img = np.full((100, 100, 3), fill_value=(i+1)*50, dtype=np.uint8)

        # save to images
        cv2.imwrite(str(images_dir / filename), img)

        # similar graph to breast_mask & dense_mask
        mask = np.full((100, 100), fill_value=50*(i+1), dtype=np.uint8)
        cv2.imwrite(str(breast_dir / filename), mask)
        cv2.imwrite(str(dense_dir  / filename), mask)

    return str(tmp_path)  # return path

def test_mammo_dataset_init(fake_dataset_root):
    """
    test if path and img nums match
    """
    dataset = MammoDataset(root_dir=fake_dataset_root, files=None, augmentations=False)
    assert len(dataset) == 3, f"Expected dataset length=3, got {len(dataset)}"

def test_mammo_dataset_getitem(fake_dataset_root):
    """
    test __getitem__ : test ability of transform img to tensor
    """
    dataset = MammoDataset(root_dir=fake_dataset_root, files=None, augmentations=False)
    
    image_tensor, breast_tensor, dense_tensor = dataset[0]

    # test shape
    assert image_tensor.shape == (3, 256, 256), f"Image tensor shape mismatch: {image_tensor.shape}"
    assert breast_tensor.shape == (1, 256, 256), f"Mask tensor shape mismatch: {breast_tensor.shape}"
    assert dense_tensor.shape == (1, 256, 256), f"Dense tensor shape mismatch: {dense_tensor.shape}"

def test_mammo_dataset_file_not_found(fake_dataset_root):
    """
    check if the file is exist
    """
    # delete one random img file
    broken_file = Path(fake_dataset_root) / "images" / "test_1.png"
    if broken_file.exists():
        broken_file.unlink()  # file deleted

    # now will raise file not finding error
    dataset = MammoDataset(root_dir=fake_dataset_root, files=None, augmentations=False)
    with pytest.raises(FileNotFoundError, match="Failed to load image"):
        _ = dataset[1]

def test_mammo_dataset_with_augmentations(fake_dataset_root):
    """
    test the function when using data enhancement
    """
    dataset = MammoDataset(root_dir=fake_dataset_root, files=None, augmentations=True)
    image_tensor, breast_tensor, dense_tensor = dataset[0]

    # basic check
    assert image_tensor.shape[1:] == (256, 256), "Image shape mismatch after augmentation + resize"
    # colorful img => shape (3, 256, 256)；mask => (1, 256, 256)

