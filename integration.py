import eel
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import joblib
import numpy as np
import Deepdensity
import segmentation_models_multi_tasking as smp
from Data_processing import load_file, convert_rgb
from CNN.Deploy import load_image, predict_masks, save_results, compute_density_ensemble
from Asymmetry_check import general_asymmetry_check, quadrant_asymmetry_check
from Quadrant_densities import quadrant_densities

"""
This code has the function which integrates all the codes together. 
It takes the files we want to upload as input (either 1 or 2 files)
It outputs the breast density, the quadrant densities, the breast asymmetry
and the quadrant asymmetry. 
This code does not include the frontend. This shows that the integration works 
in case the frontend ends up not working.
"""


def processing_image(
    uploaded_files,
):  # uploaded_files will be considered a list since we can have 1 or 2 files
    # We use empty lists so we can append the results if uploaded_filed length = 2
    Breast_density = []
    Quad_densities = []

    # Loop through uploaded files in case we upload 2 files
    for uploaded_file in uploaded_files:
        # First 2 functions from Data_processing.py to turn image pixel data to RGB
        file, image = load_file(uploaded_file)
        rgb_image = convert_rgb(file, image)

        # Lines from Deploy.py

        # CNN model weights
        model_path = Path(r"artifacts/model.pth")

        # linear regression & random forest & CatBoost: trained models
        linreg_path = Path("artifacts/linreg_model.pkl")
        rf_path = Path("artifacts/rf_model.pkl")
        catboost_path = Path("artifacts/catboost_model.pkl")

        # result saved path
        save_dir = Path("outputs")

        # CNN load an segmenting: align with training structure

        model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=None,
            classes=1,
            activation="sigmoid",
        )
        model = torch.load(model_path, map_location=torch.device("cpu"))
        model.eval()

        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        # pre-processing n predicting
        preprocess = transforms.Compose(
            [
                transforms.Resize((256, 256)),  # align with training's size
                transforms.ToTensor(),
            ]
        )

        # Start processing through the CNN (Deploy.py functions)
        img_tensor = load_image(rgb_image)
        breast_pred, dense_pred = predict_masks(img_tensor, model)
        breast_path, dense_path = save_results(
            rgb_image, breast_pred, dense_pred, save_dir
        )

        linreg_model = joblib.load(linreg_path)
        rf_model = joblib.load(rf_path)
        catboost_model = joblib.load(catboost_path)
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

        # Get the quadrant densities from Quadrant_densities.py
        q_densities = quadrant_densities(breast_path, dense_path)

        # Append the results into the list so we can save them
        Breast_density.append(final_density)
        Quad_densities.append(q_densities)

    # Checking for asymmetry if 2 images are uploaded using Asymmetry_check
    if len(uploaded_files) == 2:
        breast_asymmetry = general_asymmetry_check(Breast_density[0], Breast_density[1])
        quadrant_asymmetry = quadrant_asymmetry_check(
            Quad_densities[0], Quad_densities[1]
        )
    else:
        breast_asymmetry = "na"
        quadrant_asymmetry = "na"

    return (Breast_density, Quad_densities, breast_asymmetry, quadrant_asymmetry)


if __name__ == "__main__":  #
    # Paths can be changed accordingly
    path = Path(r"dicom.dcm")
    path2 = Path(r"dicom2.dcm")

    Breast_density, Quad_densities, breast_asymmetry, quadrant_asymmetry = (
        processing_image([path])
    )
    Breast_density1, Quad_densities1, breast_asymmetry1, quadrant_asymmetry1 = (
        processing_image([path, path2])
    )
    print(Breast_density, Quad_densities, breast_asymmetry, quadrant_asymmetry)
    print(Breast_density1, Quad_densities1, breast_asymmetry1, quadrant_asymmetry1)
