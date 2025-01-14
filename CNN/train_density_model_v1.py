# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 20:09:30 2025

@author: 27187
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import joblib

IMG_DIR       = r"F:\Leslie\uni\Y3\Programming\project\main_dataset\train\images"
BREAST_DIR    = r"F:\Leslie\uni\Y3\Programming\project\main_dataset\train\breast_masks"
DENSE_DIR     = r"F:\Leslie\uni\Y3\Programming\project\main_dataset\train\dense_masks"
CSV_PATH      = r"F:\Leslie\uni\Y3\Programming\project\main_dataset\train_files_processed.csv"
THRESHOLD     = 128
TEST_RATIO    = 0.1   # use 10% of data as val

"""
this code is for training the three models(linear regression, random forest, catboost) for supporting predicting the density
the predeiction was weighted optimized, to maximize the accuracy.
"""

def check_file_exists(path, descr="file"):
    #check the existence of file
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{descr} not found: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"{descr} is empty (0 bytes): {path}")


def compute_area(mask_path, threshold=128):
    """load single channel img，according to threshold to calc area after binarized"""
    if threshold < 0 or threshold > 255:
        raise ValueError(f"Threshold must be in [0,255], got {threshold}")

    # load img
    mask = Image.open(mask_path).convert('L')
    mask_np = np.array(mask)
    # binarized
    mask_bin = (mask_np > threshold).astype(np.uint8)
    return np.sum(mask_bin)


try:
    # check is csv file existed
    check_file_exists(CSV_PATH, "CSV file")

    # read CSV
    df = pd.read_csv(CSV_PATH)
    print("CSV shape:", df.shape)

    # feature and path def
    breast_areas = []
    dense_areas  = []
    density_list = []
    
    # walk through csv
    for i, row in df.iterrows():
        # row 0: filename; row 1: density
        if len(row) < 2:
            raise ValueError(f"CSV row has insufficient columns at index {i}: {row}")

        filename     = row.iloc[0]
        true_density = row.iloc[1]

        # def mask path
        breast_path = os.path.join(BREAST_DIR, filename)
        dense_path  = os.path.join(DENSE_DIR,  filename)

        # check is mask existed
        check_file_exists(breast_path, "breast mask")
        check_file_exists(dense_path,  "dense mask")

        # calc area
        breast_area = compute_area(breast_path, THRESHOLD)
        dense_area  = compute_area(dense_path,  THRESHOLD)

        breast_areas.append(breast_area)
        dense_areas.append(dense_area)
        density_list.append(true_density)

    # calc ratio = dense / breast
    ratios = []
    for b, d in zip(breast_areas, dense_areas):
        if b == 0:
            ratios.append(0.0)  # 0 division exception
        else:
            ratios.append(d / b)

    # def eigen matrix
    X = np.column_stack((breast_areas, dense_areas, ratios))  # shape [N,3]
    y = np.array(density_list)                                # shape [N,]

    if len(X) == 0:
        raise ValueError("No data loaded to train the models. Check CSV or masks folder.")
    if len(X) != len(y):
        raise ValueError(f"Mismatch: X has {len(X)} samples, y has {len(y)} samples.")

    # Train/Test splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=42
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # train the three models: LinearRegression, RandomForest, CatBoost
    print("Training models...")
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)

    rf = RandomForestRegressor(max_depth=5, random_state=0)
    rf.fit(X_train, y_train)

    catboost = CatBoostRegressor(iterations=300, random_seed=42, verbose=False)
    catboost.fit(X_train, y_train)

    # model evaluation
    print("LinearRegression R2:", linreg.score(X_test, y_test))
    print("RandomForest    R2:", rf.score(X_test, y_test))
    print("CatBoost       R2:", catboost.score(X_test, y_test))

    # model saved
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(linreg,   os.path.join(save_dir, "linreg_model.pkl"))
    joblib.dump(rf,       os.path.join(save_dir, "rf_model.pkl"))
    joblib.dump(catboost, os.path.join(save_dir, "catboost_model.pkl"))
    print(f"All three models saved in '{save_dir}' directory.")

# overall common exceptions
except FileNotFoundError as e:
    print(f"Error: {e}")
except ValueError as e:
    print(f"ValueError: {e}")
except RuntimeError as e:
    print(f"RuntimeError: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
