# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 19:33:48 2025

@author: 27187
"""

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import joblib
import sys
from pathlib import Path

import segmentation_models_multi_tasking as smp

# use GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# transforms
preprocess = transforms.Compose(
    [
        transforms.Resize((256, 256)),  # align with training's size
        transforms.ToTensor(),
    ]
)


# error handling: check file existence
def check_file_exists(path, descr="file"):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{descr} not found at: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"{descr} is empty (0 bytes): {path}")


def load_image(image_path):
    pil_img = Image.fromarray(image_path)
    img_tensor = preprocess(pil_img).unsqueeze(0)  # [1,3,256,256]
    img_tensor = img_tensor.to("cpu")
    return img_tensor


def predict_masks(img_tensor, model):
    with torch.no_grad():
        output = model(img_tensor)
    # check output shape
    if not isinstance(output, tuple):
        raise ValueError(
            "Model output is not a tuple (breast, dense). "
            "Check your model definition or set classes=2 and output as a tuple."
        )
    if len(output) != 2:
        raise ValueError(
            f"Expected model output of length 2, got length {len(output)}. "
            "Should be (breast_pred, dense_pred)."
        )
    breast_tensor, dense_tensor = output
    # convert to CPU numpy
    breast_pred = breast_tensor[0, 0].cpu().numpy()
    dense_pred = dense_tensor[0, 0].cpu().numpy()
    return breast_pred, dense_pred


def save_results(img_path, breast_pred, dense_pred, save_directory):
    """
    Save original img and predicted results
    """
    pil_img = Image.fromarray(img_path)
    os.makedirs(save_directory, exist_ok=True)

    # load original
    original_save_path = os.path.join(save_directory, "original_image.png")
    pil_img.save(original_save_path)
    print(f"Original image saved => {original_save_path}")

    # 2) breast mask
    breast_img = Image.fromarray((breast_pred * 255).astype(np.uint8)).convert("L")
    breast_path = os.path.join(save_directory, "prediction_breast.png")
    breast_img.save(breast_path)
    print(f"Breast mask saved => {breast_path}")

    # 3) dense mask
    dense_img = Image.fromarray((dense_pred * 255).astype(np.uint8)).convert("L")
    dense_path = os.path.join(save_directory, "prediction_dense.png")
    dense_img.save(dense_path)
    print(f"Dense mask saved => {dense_path}")

    return breast_path, dense_path


# load 3 ML models with error handling
def load_ml_model(path, name="ML model"):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"{name} is empty: {path}")
    return joblib.load(path)


def compute_density_ensemble(
    breast_mask, dense_mask, linreg, rf, catboost, threshold=128
):
    """
    Params
    ------
    breast_mask, dense_mask: np.array (0~255)
    """
    breast_bin = (breast_mask > threshold).astype(np.uint8)
    dense_bin = (dense_mask > threshold).astype(np.uint8)

    breast_area = np.sum(breast_bin)
    dense_area = np.sum(dense_bin)
    if breast_area == 0:
        # handle zero-division or no-breast case
        return 0.0

    ratio = dense_area / breast_area

    X_input = np.array([[breast_area, dense_area, ratio]])
    # individual predictions
    pred_lin = linreg.predict(X_input)[0]
    pred_rf = rf.predict(X_input)[0]
    pred_cat = catboost.predict(X_input)[0]

    # 4) weight fusion
    combined_density = 0.4 * pred_lin + 0.3 * pred_rf + 0.3 * pred_cat
    return combined_density


if __name__ == "__main__":
    # CNN model weights
    model_path = Path(r"artifacts/model.pth")

    # linear regression & random forest & CatBoost: trained models
    linreg_path = Path("artifacts/linreg_model.pkl")
    rf_path = Path("artifacts/rf_model.pkl")
    catboost_path = Path("artifacts/catboost_model.pkl")
    # input image
    image_path = r"F:\Leslie\uni\Y3\Programming\project\main_dataset\train\images\03fd56d2b8ad85b9f1eddc9d07fa5c3c.jpg"

    # result saved path
    save_dir = Path("outputs")

    print("Checking file paths...")

    # Check if segmentation model weights exist
    check_file_exists(model_path, "Segmentation model weights")

    # Check if ML models exist
    check_file_exists(linreg_path, "Linear Regression model")
    check_file_exists(rf_path, "Random Forest model")
    check_file_exists(catboost_path, "CatBoost model")

    # Check if image file exist
    check_file_exists(image_path, "Input image")

    print("All files are valid.")

    # load segmentation model
    print("Loading segmentation model...")
    try:
        # create SMP UNet structure
        seg_model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=None,
            classes=1,
            activation="sigmoid",
        )
        # load weights
        seg_model = torch.load(model_path, map_location=DEVICE)
        seg_model = seg_model.to(DEVICE)
        seg_model.eval()
        print("Segmentation model loaded successfully.")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("Error: CUDA out of memory while loading the segmentation model.")
        else:
            print(f"RuntimeError during model loading: {e}")
        raise  # re-raise if you want to stop execution
    except Exception as e:
        print(f"Unexpected error during model loading: {e}")
        raise

    print("Loading 3 ML models: LinearRegression, RandomForest, CatBoost...")

    try:
        linreg_model = load_ml_model(linreg_path, "Linear Regression")
        rf_model = load_ml_model(rf_path, "Random Forest")
        catboost_model = load_ml_model(catboost_path, "CatBoost")
        print("All 3 ML models loaded successfully.")
    except Exception as e:
        print(f"Error loading ML models: {e}")
        raise

    try:
        print("\n=== Step1: Segmentation Inference ===")
        breast_pred, dense_pred = predict_masks(image_path, seg_model)

        print("\n=== Step2: Save segmentation results ===")
        breast_path, dense_path = save_results(
            image_path, breast_pred, dense_pred, save_dir
        )

        print("\n=== Step3: Compute final density ===")
        # read back
        breast_mask_arr = np.array(Image.open(breast_path).convert("L"))
        dense_mask_arr = np.array(Image.open(dense_path).convert("L"))

        final_density = compute_density_ensemble(
            breast_mask_arr,
            dense_mask_arr,
            linreg_model,
            rf_model,
            catboost_model,
            threshold=128,
        )
        print(f"\n** Final Combined Density: {final_density:.2f}% **")

    except KeyboardInterrupt:
        print("\nUser interrupted the process manually. Exiting gracefully...")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(
                "Error: CUDA out of memory during inference. Reduce image/batch size."
            )
        else:
            print(f"RuntimeError: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

