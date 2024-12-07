import pathlib
import torch
from torchvision import transforms
from PIL import Image
from models import MyModel
from SimpleYOLO import SimpleYOLO
import yaml
from data_processing_cifar import DataProcessorCIFAR
from data_processing_kitti import DataProcessorKitti
from load_args import LoadArgs
import matplotlib.pyplot as plt
from process_output_img import ProcessOutputImg

# Load arguments from config file
kwargs = LoadArgs()
batch_size = kwargs.pop("batch_size",1)
model_type = kwargs.pop("model_type", "linear")
data_type = kwargs.pop("data_type", "cifar")
training_split_percentage = kwargs.pop("training_split_percentage", 0.8)
dataset_percentage = kwargs.pop("dataset_percentage", 1.0)

# Load model
if model_type == "simpleYOLO":
  model = SimpleYOLO()
else:
  model = MyModel(model_type, batch_size)

# Load weights from trained model and set model in eval mode
basedir = pathlib.Path(__file__).parent.parent.parent.resolve()
state_dict = torch.load(str(basedir) + "/models/06_12-22:23:37/simpleyolo_loss_0.764_epoch_38.pt")
model.load_state_dict(state_dict)
model.eval()

# Load testing data
if data_type == "cifar":
  _,_,test_dataset = DataProcessorCIFAR(batch_size)
elif data_type == "kitti":
  _,_,test_dataset = DataProcessorKitti(batch_size, training_split_percentage=0.8, dataset_percentage=1.0)

# This is grabbing an input image from the directory
# image_path = str(basedir) + "/src/custom/input_media/bird.jpg"
# input_rgb = Image.open(image_path).convert("RGB")
# input_tensor = preprocess(input_rgb).unsqueeze(0)

img = test_dataset[0][0]
truth = test_dataset[0][1]
input_tensor = img.unsqueeze(0)

# Run inference
with torch.no_grad():
  output = model(input_tensor)
  
# Process output kitti
ProcessOutputImg(img, output, truth)

test=1
  
# Process output
# probs = torch.nn.functional.softmax(output[0],dim=0)
# prediction = probs.argmax().item()

# classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
#                    'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# print(f'We are {round(probs[prediction].item()*100,2)}% sure this is a(n) {classes[prediction]}')