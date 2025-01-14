# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 00:26:04 2025

@author: 27187
"""

import os
import pytest
import subprocess
import numpy as np
import cv2

@pytest.fixture
def prepare_fake_env(tmp_path):
    """
    test in pytest's'tmp_path:
    - images, breast_masks, dense_masks sub_folder
    - train_files_processed.csv
    - create test img & CSV 
    return: (root_dir, csv_path, python_script_path)
    """
    # create sub_folder
    images_dir = tmp_path / "images"
    breast_dir = tmp_path / "breast_masks"
    dense_dir  = tmp_path / "dense_masks"

    images_dir.mkdir()
    breast_dir.mkdir()
    dense_dir.mkdir()

    # create test img
    num_samples = 3
    filenames = []
    for i in range(num_samples):
        filename = f"test_{i}.png"
        filenames.append(filename)

        # create 100x100 3-channel img
        img = np.full((100, 100, 3), fill_value=(i+1)*40, dtype=np.uint8)
        cv2.imwrite(str(images_dir / filename), img)

        # breast & dense is single channel grey scale img
        mask = np.full((100, 100), fill_value=50*(i+1), dtype=np.uint8)
        cv2.imwrite(str(breast_dir / filename), mask)
        cv2.imwrite(str(dense_dir  / filename), mask)

    # create a smallest csv file
    csv_path = tmp_path / "train_files_processed.csv"
    with csv_path.open("w") as f:
        f.write("filename,density\n")
        for i, fn in enumerate(filenames):
            f.write(f"{fn},{0.1*(i+1)}\n")

    python_script_path = r"F:\Leslie\uni\Y3\Programming\project\Repo\cancer_masking\CNN\newest_version\train_density_model_v1.py"

    return (tmp_path, csv_path, python_script_path)

def test_train_density_models_ok(prepare_fake_env):
  
    root_dir, csv_path, script_path = prepare_fake_env

    # modify the path in script
    env = os.environ.copy()
    env["F_LESLIE_IMG_DIR"]    = str(root_dir / "images")
    env["F_LESLIE_BREAST_DIR"] = str(root_dir / "breast_masks")
    env["F_LESLIE_DENSE_DIR"]  = str(root_dir / "dense_masks")
    env["F_LESLIE_CSV_PATH"]   = str(csv_path)

    ret = subprocess.run(
        ["python", str(script_path)],
        env=env,
        capture_output=True,
        text=True
    )

    print("STDOUT:", ret.stdout)
    print("STDERR:", ret.stderr)

    # check if script exit with 0
    assert ret.returncode == 0, f"Script crashed with code {ret.returncode}"

    # check if pkl existed
    saved_models_dir = os.path.join(os.getcwd(), "saved_models")

    assert os.path.exists(os.path.join(saved_models_dir, "linreg_model.pkl")), "linreg_model.pkl not found"
    assert os.path.exists(os.path.join(saved_models_dir, "rf_model.pkl")),     "rf_model.pkl not found"
    assert os.path.exists(os.path.join(saved_models_dir, "catboost_model.pkl")),"catboost_model.pkl not found"
