# Autonomous Vehicle Object Detection
The purpose of this project is to develop a custom road obstacle detection with bounding boxes CNN model and comapre it to state-of-the-art models like YOLO.
## File Structure:
- `src`: Source code
  - `.py`: Core function for training the model, called from `run.py`
  - `model.py`: NN model definition
  - `run.py`: Generates inputs for the model and runs the training
  - `loss.py`: Loss function
- `config.yaml`: Model training parameters
- `models`: Saved models
