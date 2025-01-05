# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 05:30:48 2025

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

# function calc mask area
def compute_area(mask_path, threshold=128):

    mask = Image.open(mask_path).convert('L')
    mask_np = np.array(mask)
    mask_bin = (mask_np > threshold).astype(np.uint8)
    return np.sum(mask_bin)

# 1) read density file
df = pd.read_csv(CSV_PATH)
print("CSV shape:", df.shape)

# 2) load file name and density from .csv
breast_areas = []
dense_areas  = []
density_list = []

for i, row in df.iterrows():
    filename     = row.iloc[0]   # rwo 0: filename
    true_density = row.iloc[1]   # row 1: density value

    # calc breast_area, dense_area
    breast_path = os.path.join(BREAST_DIR, filename)
    dense_path  = os.path.join(DENSE_DIR,  filename)

    breast_area = compute_area(breast_path, THRESHOLD)
    dense_area  = compute_area(dense_path,  THRESHOLD)

    breast_areas.append(breast_area)
    dense_areas.append(dense_area)
    density_list.append(true_density)

# 3) calculating ratio = dense / breast
ratios = []
for b, d in zip(breast_areas, dense_areas):
    if b == 0:
        ratios.append(0.0)  # 0 division error handling
    else:
        ratios.append(d / b)

# 4) fused feature
X = np.column_stack((breast_areas, dense_areas, ratios))  # shape [N,3]
y = np.array(density_list)                                # shape [N,]

# 5) Train/Test dividing with ratio : 90%: 10%
X_train, X_test, y_train, y_test = train_test_split(X, y, 
    test_size=TEST_RATIO, random_state=42)

print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# 6) train three ML model: linear regression, random forest, Catboost
linreg = LinearRegression()
linreg.fit(X_train, y_train)

rf = RandomForestRegressor(max_depth=5, random_state=0)
rf.fit(X_train, y_train)

catboost = CatBoostRegressor(iterations=300, random_seed=42, verbose=False)
catboost.fit(X_train, y_train)

# 7) model evaluating
print("LinearRegression R2:", linreg.score(X_test, y_test))
print("RandomForest    R2:", rf.score(X_test, y_test))
print("CatBoost       R2:", catboost.score(X_test, y_test))

# 8) model saved
os.makedirs("saved_models", exist_ok=True)
joblib.dump(linreg,   "saved_models/linreg_model.pkl")
joblib.dump(rf,       "saved_models/rf_model.pkl")
joblib.dump(catboost, "saved_models/catboost_model.pkl")
print("All three models saved in 'saved_models' directory.")

