# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 21:35:56 2024

@author: 27187
"""

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import joblib
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import segmentation_models_multi_tasking as smp


def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0)  # [1,3,256,256]
    return img_tensor.to(DEVICE)


def predict_masks(image_path, model):
    img_tensor = load_image(image_path)
    with torch.no_grad():
        output = model(img_tensor)
    if not isinstance(output, tuple):
        raise ValueError(
            "Model output is not a tuple (breast, dense). Check your model definition."
        )
    breast_tensor, dense_tensor = output
    breast_pred = breast_tensor[0, 0].cpu().numpy()
    dense_pred = dense_tensor[0, 0].cpu().numpy()
    return breast_pred, dense_pred


def save_results(image_path, breast_pred, dense_pred, save_dir):
    """
    saving original img and predicted result
    """
    img = Image.open(image_path).convert("RGB")
    os.makedirs(save_dir, exist_ok=True)

    # 1) original img
    original_save_path = os.path.join(save_dir, "original_image.png")
    img.save(original_save_path)
    print(f"Original image saved => {original_save_path}")

    # 2) breast mask
    breast_img = Image.fromarray((breast_pred * 255).astype(np.uint8)).convert("L")
    breast_path = os.path.join(save_dir, "prediction_breast.png")
    breast_img.save(breast_path)
    print(f"Breast mask saved => {breast_path}")

    # 3) dense mask
    dense_img = Image.fromarray((dense_pred * 255).astype(np.uint8)).convert("L")
    dense_path = os.path.join(save_dir, "prediction_dense.png")
    dense_img.save(dense_path)
    print(f"Dense mask saved => {dense_path}")

    return breast_path, dense_path


# define dense calculating function combining with ML models to improve accuracy
def compute_density_ensemble(
    breast_mask, dense_mask, linreg, rf, catboost, threshold=128
):
    """
    Parameters
    ----------
    breast_mask, dense_mask : np.array, 0~255
    linreg, rf, catboost    : ML models
    Returns
    -------
    combined_density : float, processed prediction density
    """
    # 1) binarized
    breast_bin = (breast_mask > threshold).astype(np.uint8)
    dense_bin = (dense_mask > threshold).astype(np.uint8)

    # 2) area
    breast_area = np.sum(breast_bin)
    dense_area = np.sum(dense_bin)
    if breast_area == 0:
        return 0.0
    ratio = dense_area / breast_area

    # 3) predicting by ML models
    X_input = np.array([[breast_area, dense_area, ratio]])
    pred_lin = linreg.predict(X_input)[0]
    pred_rf = rf.predict(X_input)[0]
    pred_cat = catboost.predict(X_input)[0]

    # 4) weight fusion
    combined_density = 0.4 * pred_lin + 0.3 * pred_rf + 0.3 * pred_cat
    return combined_density


if __name__ == "__main__":
    # use GPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CNN model weights
    model_path = r"C:\Users\layan\OneDrive\Desktop\Breast cancer project\cancer_masking\result_epoch{epoch + 1}.pth"

    # linear regression & random forest & CatBoost: trained models
    linreg_path = r"C:\Users\layan\OneDrive\Desktop\pkl_files\linreg_model.pkl"
    rf_path = r"C:\Users\layan\OneDrive\Desktop\pkl_files\rf_model.pkl"
    catboost_path = r"C:\Users\layan\OneDrive\Desktop\pkl_files\catboost_model.pkl"

    # input image
    # image_path = r"F:\Leslie\uni\Y3\Programming\project\main_dataset\train\images\03fd56d2b8ad85b9f1eddc9d07fa5c3c.jpg"

    # result saved path
    save_dir = r"C:\Users\layan\OneDrive\Desktop\save_dir"

    # CNN load an segmenting: align with training structure
    print("Loading segmentation model...")
    model = smp.Unet(
        encoder_name="resnet50", encoder_weights=None, classes=1, activation="sigmoid"
    )
    model = torch.load(model_path, map_location=DEVICE)
    model = model.to(DEVICE)
    model.eval()
    print("Segmentation model loaded.")

    # pre-processing n predicting
    preprocess = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # align with training's size
            transforms.ToTensor(),
        ]
    )

    # load density calculating models
    print("Loading 3 ML models: LinearRegression, RandomForest, CatBoost...")
    linreg_model = joblib.load(linreg_path)
    rf_model = joblib.load(rf_path)
    catboost_model = joblib.load(catboost_path)
    print("All 3 models loaded.")
