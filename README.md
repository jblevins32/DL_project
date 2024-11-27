# Autonomous Vehicle Object Detection
The purpose of this project is to develop a custom road obstacle detection with bounding boxes CNN model and compare it to state-of-the-art models like YOLO.

## File Structure:
- `data`: Training data
- `figs`: Custom training output figures
- `models`: Saved models, either prebuilt yolo or our own generated models
- `src`: Source code
  - `custom`: Custom model training and inference
    - `input media`: Place input media here to feed to the model
    - `data_processing.py`: Extracts data from the chosen dataset and sets it up for training 
    - `inference.py`: Run inference on the trained model
    - `models.py`: NN model definitions. Here you can make models and choose which to utilize
    - `solver.py`: Core function for training the model, called from `run.py`
    - `train.py`: Loads hyperparameters for the model and runs the training
  - `YOLO`: YOLO training and inference
    - `input media`: Place input media here to feed to the model
    - `output data`: Place output data here from the model
    - `output media`: Place output media here from the model
    - `get_data.py`: Gets bounding box info extracted from the YOLO model
    - `yolo_world.py`: Main inference of YOLO models
- `config.yaml`: Model training hyperparameters
- `environment.yaml`: Conda environment
