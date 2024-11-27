import pathlib
import torch
from torchvision import transforms
from PIL import Image
from models import MyModel
import yaml

# Load a chosen model
with open("config.yaml", "r") as read_file:
  config = yaml.safe_load(read_file)
model_type = config['Train']['model_type']
batch_size = config['Train']['batch_size']

model = MyModel(model_type, batch_size)

# Load weights from trained model and set model in eval mode
basedir = pathlib.Path(__file__).parent.parent.parent.resolve()
state_dict = torch.load(str(basedir) + "/models/customcnn_loss_0.567.pt")
model.load_state_dict(state_dict)
model.eval()

# Load input image and process
preprocess = transforms.Compose([
  transforms.Resize((32,32)), # This image size is currently set to the image size of the cifar10 dataset
  transforms.ToTensor(),
  transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010])
])

image_path = str(basedir) + "/src/custom/input_media/bird.jpg"
input_rgb = Image.open(image_path).convert("RGB")
input_tensor = preprocess(input_rgb).unsqueeze(0)

# Run inference
with torch.no_grad():
  output = model(input_tensor)
  
# Process output
probs = torch.nn.functional.softmax(output[0],dim=0)
prediction = probs.argmax().item()

classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
                   'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

print(f'We are {round(probs[prediction].item()*100,2)}% sure this is a(n) {classes[prediction]}')