import eel
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import joblib
import numpy as np
import segmentation_models_multi_tasking as smp
from Data_processing import load_file, convert_rgb
from CNN.Deploy import load_image, predict_masks, save_results, compute_density_ensemble
from qudrantdensities import background_clean, quadrant_densities

eel.init("Frontend")  # initialising our directory
eel.start("index.html")


@eel.expose
def processing_image(
    uploaded_files,
):  # uploaded_files will be considered a list since we can have 1 or 2 files
    for uploaded_file in uploaded_files:
        file, image = load_file(uploaded_file)
        rgb_image = convert_rgb(file, image)

        # use GPU
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        model = torch.load(model_path, map_location=DEVICE)
        model = model.to(DEVICE)
        model.eval()

        # pre-processing n predicting
        preprocess = transforms.Compose(
            [
                transforms.Resize((256, 256)),  # align with training's size
                transforms.ToTensor(),
            ]
        )

        breast_pred, dense_pred = predict_masks(rgb_image, model)
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
        # Some elements needed in deploy are outside the function: gotta figure out if i should
        # copy the lines or if there is another way to incorporate it.
        masked_dense = background_clean(breast_path, dense_path)
        quadrant_densities = quadrant_densities(breast_path, dense_path)
        result = (
            f"Breast density: {final_density}, Quadrant densities: {quadrant_densities}"
        )

    return result


# @eel.expose and then you define the function
# To call: eel.expose(function_in_java)
