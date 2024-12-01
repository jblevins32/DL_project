# Autonomous Vehicle Object Detection
The purpose of this project is to improve upon current state of the art object detection models by creating a vision-based perception model that is capable of identifying and bboxing common road obstacles from visual data (cars, people, cyclists, etc.).

## TO DO:
- Implement [KITTI](https://www.cvlibs.net/datasets/kitti/raw_data.php), BDD100K, [Waymo](https://waymo.com/open/download/), or ApolloScape datasets
- Add functionality to train models with bounding boxes
- Modify YOLO to train on new data
- Implement other models: Faster R-CNN, Mask R-CNN, EfficientDet
- Implement visualization to compare results
### Stretch goals:
- Context awareness with attention or RNNs
- Add depth estimation
- Apply to phone app (may need to optimize for edge devices)

## How to Use:
- Training custom model:
  - Adjust hyperparameters in `config.yaml`
  - Run `train.py`
  - Note: This is currently running on CIFAR10 dataset and will need to be generalized for otherdatasets.
- Inference custom model:
  - Run `inference_custom.py`. Be sure to adjust the input images to infer on.
- Inference YOLO:
  - Run `inference_yolo.py`. Be sure to adjust the input images to infer on.

## Notes:
- Some of the files in the below directory are not showing up. This is because they are in the .gitignore because they have too much data to push. Because of this, some models such as yolov8s-worldv2.pt will need to be downloaded.

## File Structure:
- `data`: Training data
- `figs`: Custom training output figures
- `models`: Saved models, either prebuilt or our own generated models
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
    - `inference_yolo.py`: Main inference of YOLO models
- `config.yaml`: Model training hyperparameters
- `environment.yaml`: Conda environment
