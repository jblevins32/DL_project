import pathlib
from models import MyModel
from SimpleYOLO import SimpleYOLO
from AttentionYOLO import AttentionYOLO
from EncoderDecoderYOLO import EncoderDecoderYOLO
from tinyYOLO import TinyYOLO
from data_processing_cifar import DataProcessorCIFAR
from data_processing_kitti import DataProcessorKitti
from load_args import LoadArgs
from process_output_img import *

# Load arguments from config file
kwargs = LoadArgs()
batch_size = kwargs.pop("batch_size",1)
model_type = kwargs.pop("model_type", "linear")
data_type = kwargs.pop("data_type", "cifar")
training_split_percentage = kwargs.pop("training_split_percentage", 0.8)
dataset_percentage = kwargs.pop("dataset_percentage", 1.0)
num_classes = kwargs.pop("num_classes", 4)

# Load model
if model_type == "simpleYOLO":
  model = SimpleYOLO(num_classes=num_classes)
elif model_type == "attentionYOLO":
  model = AttentionYOLO(num_classes=num_classes)
elif model_type == "EncoderDecoderYOLO":
  model = EncoderDecoderYOLO(num_classes=num_classes)
elif model_type == "tinyYOLO":
  model = TinyYOLO(num_classes=num_classes)
else:
  model = MyModel(model_type, batch_size)

# Load weights from trained model and set model in eval mode
basedir = pathlib.Path(__file__).parent.parent.parent.resolve()
state_dict = torch.load(str(basedir) + "/models/encoderdecoder/encoderdecoderyolo_loss_0.958_f1_0.6917_epoch_66.pt") 
# best_simpleyolo_correctedloss/simpleyolo_loss_0.304_f1_0.8261_epoch_92.pt"
# best_tinyyolo/simpleyolo_loss_2.605_f1_0.6142_epoch_111.pt
# encoderdecoder/encoderdecoderyolo_loss_0.958_f1_0.6917_epoch_66.pt
model.load_state_dict(state_dict)
model.eval()

# Load testing data
if data_type == "cifar":
  _,_,test_dataset = DataProcessorCIFAR(batch_size)
elif data_type == "kitti":
  _,_,test_dataset = DataProcessorKitti(batch_size, training_split_percentage, dataset_percentage)

# This is grabbing an input image from the directory
# image_path = str(basedir) + "/src/custom/input_media/bird.jpg"
# input_rgb = Image.open(image_path).convert("RGB")
# input_tensor = preprocess(input_rgb).unsqueeze(0)

# Set here to test specific frame
image_idx = 15

img = test_dataset[image_idx][0]
label = test_dataset[image_idx][1] # (n,5) = number of boxes, 4 bbox coords and 1 class ID
input_tensor = img.unsqueeze(0)

# Run inference
with torch.no_grad():
  output = model(input_tensor)
  
# Process output kitti
# ShowResults(img, output)
predictions = ProcessOutputImg(img, output, label, num_classes = num_classes)



test=1
  
# Process output cifar input image from the directory
# probs = torch.nn.functional.softmax(output[0],dim=0)
# prediction = probs.argmax().item()

# classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
#                    'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# print(f'We are {round(probs[prediction].item()*100,2)}% sure this is a(n) {classes[prediction]}')