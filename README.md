In order for the code to be run, some elements have to be in the directory, that could not be uploaded on the Githubb main branch:

- An empty folder called "outputs"
- A folder called "artifacts" with files called "model.pth", "linreg_model.pkl", "rf_model.pkl" and "catboost_model.pkl". This folder will be provided
- A folder named "Deepdensity". It will also be sent
- The folders are all on a folder named "Synapse_Systems". Make sure to remove them from that folder and into the directory when you want to run the code

The code that should be run in main.py
The folder that will be sent will be shared via Outlook to m.holloway@imperial.ac.uk
Or access via: https://imperiallondon-my.sharepoint.com/:f:/g/personal/lm1122_ic_ac_uk/EgmhZ6aQ8oZPhssz0uricXIB7tY7Y46soJuZUtjlftoQaA?e=IMIVMk 

If main.py does not work (integration does not work with frontend), also run integration.py to see that the function works.

For the CNN training part:
## File Structure
- `train_v3_loss.py`:
  - Trains the segmentation model.
  - Logs metrics such as accuracy, IoU, and F1-score.

- `deploy_v2.py`:
  - Performs deployment using the trained segmentation model and trained density prediction model.
  - Saves segmented masks and computes breast density using ensemble models.

- `mammo_dataset_v1.py`:
  - Custom dataset class for loading mammogram images and corresponding masks.
  - Includes data augmentation and tensor transformation.

- `loss_v1.py`:
  - Custom BCE-Dice Loss function for segmentation tasks.
  - Includes error handling for tensor dimensions and value ranges.

- `train_density_model_models.py`:
  - Prepares training data from segmentation results.
  - Trains Linear Regression, Random Forest, and CatBoost models for density estimation.
  - Saves trained models for deployment.
    
## Acknowledgments

The train_v3_loss.py uses code from the [Deepdensity]([https://github.com/USERNAME/REPOSITORY](https://github.com/uefcancer/Deepdensity/tree/main)), licensed under the [MIT License]((https://github.com/uefcancer/Deepdensity/blob/main/LICENSE)).

Copyright (c) 2023 uefcancer.
 
