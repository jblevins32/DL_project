# Autonomous Vehicle Object Detection
The purpose of this project is to improve upon current state of 
the art object detection models by creating a vision-based perception model 
that is capable of identifying and bboxing common road obstacles from visual data 
(cars, pedestirans, vans, cyclists, etc.).

## How to Use:
- Training custom model:
  - Adjust hyperparameters and current model type selection in `config.yaml`
  - Run `train.py`
    - Note: This is currently running on KITTI dataset and will need to be generalized for other datasets.
    - This script will check for all of the necessary files and directories to perform training.
      - If they do not exist, the dataset will automatically be downloaded to `DL_project/dataset`
  - Select model and copy its path into inference.py to use a specific trained model
  - Run `inference.py`. Be sure to adjust the input images to infer on.

## Notes:
- Some of the files in the below directory are not showing up. 
  - They are in the .gitignore because they have too much data to push.

## File Structure:
- `dataset`: All image files and corresponding labels for Training/Testing
  - `images`: Training and Testing image data
    - `training`
    - `testing`
  - `labels`: Labels for Training data, no labels for Testing data
- `figs`: Custom training output figures
- `models`: Saved models
- `src`: Source code
  - `custom`: Custom model training and inference
    - `input media`: Place input media here to feed to the model
    - `data_processing.py`: Extracts data from the chosen dataset and sets it up for training 
    - `inference.py`: Run inference on the trained model
    - `models.py`: NN model definitions. Here you can make models and choose which to utilize
    - `solver.py`: Core function for training the model, called from `run.py`
    - `train.py`: Loads hyperparameters for the model and runs the training
- `config.yaml`: Model training hyperparameters
- `environment.yaml`: Conda environment

## Conda:
- Create env from `environment.yaml`
- Still need to run `pip install torcheval` after activating conda env, pkg cannot be installed from conda env