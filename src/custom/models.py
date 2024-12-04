import torch.nn as nn
from torchvision.models import alexnet, resnet18

class MyModel(nn.Module):
    def __init__(self, model_type, batch_size):
        super(MyModel, self).__init__()
        
        # Calculate the output dimension
        self.conv_output_dim = 10 #NEED TO CHANGE
        self.batch_size = batch_size
        self.model_type = model_type
        
        # Define a simple fully connected neural network
        self.linear_input_dim = 3072 #NEED TO CHANGE
        self.linear_output_dim = 3072 #NEED TO CHANGE
        
        if self.model_type == "linear":
            drop_rate = 0.01
            
            self.model = nn.Sequential(
                nn.Linear(self.linear_input_dim, 128),   # Input layer
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                
                nn.Linear(128, 256), 
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                
                nn.Linear(512, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                
                nn.Linear(1024, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                
                nn.Linear(128, self.linear_output_dim)   # Output layer
            )

            # Print the number of training parameters in the model
            self.count_params(self.model)
            
        # Resnet18
        if self.model_type == "resnet":
            self.model = resnet18(weights=False)
            
            self.model.conv1 = nn.Conv2d(in_channels=3,
                out_channels=64,
                kernel_size=2,
                stride=1,
                padding=0,
                bias=False
            )
            
            self.model.fc = nn.Linear(512, self.conv_output_dim)
            
            # Print the number of training parameters in the model
            self.count_params(self.model)
        
        # Alexnet
        if self.model_type == "alexnet":
            self.model = alexnet(weights=False)
            
            self.model.features[0] = nn.Conv2d(in_channels=1,
                out_channels=64,
                kernel_size=2,
                stride=1,
                padding=0,
                bias=False
            )
        
            self.model.classifier[6] = nn.Linear(4096, self.conv_output_dim)
            
            # Print the number of training parameters in the model
            self.count_params(self.model)
                    
        # Custom CNN
        if self.model_type == "customcnn":
            self.flatten = nn.Flatten()
            self.model = nn.Sequential(
                
                # Conv layer, 2D becuase input is image matrix
                nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=1,padding=3),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=3,stride=2),
                            
                nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(64),
                
                nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(128),
                
                nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(256),
                
                nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(512),
                
                # nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,stride=2,padding=1),
                # nn.BatchNorm2d(1024),
                # nn.LeakyReLU(),
                # nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1,padding=1),
                # nn.BatchNorm2d(1024),
                
                nn.AdaptiveAvgPool2d((1,1)),
            )
        
            self.fully_connected = nn.Sequential(
                nn.Linear(512,out_features=10)
            )
            
        # Custom CNN based on YOLO... comments are output sizes of that layer (chanels, height, width)
        if self.model_type == "customcnn_kitti":
            self.flatten = nn.Flatten()
            self.model = nn.Sequential(
                
                # Conv layer, 2D because input is image matrix
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3), # 64x365x1220
                nn.MaxPool2d(kernel_size=2, stride=2), # 64x182x610
                
                nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1), # 192x182x610
                nn.MaxPool2d(kernel_size=2, stride=2), # 192x91x305
                
                nn.Conv2d(192, 128, kernel_size=1, stride=1), # 128x91x305
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # 256x91x305
                nn.Conv2d(256, 256, kernel_size=1, stride=1), # 256x91x305
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 512x91x305
                nn.MaxPool2d(kernel_size=2, stride=2), # 512x45x152
                
                nn.Conv2d(512, 256, kernel_size=1, stride=1), # 256x45x152
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 512x45x152
                nn.Conv2d(512, 256, kernel_size=1, stride=1), # 256x45x152
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 512x45x152
                nn.Conv2d(512, 256, kernel_size=1, stride=1), # 256x45x152
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 512x45x152
                nn.Conv2d(512, 256, kernel_size=1, stride=1), # 256x45x152
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 512x45x152
                
                nn.Conv2d(512, 512, kernel_size=1, stride=1), # 512x45x152
                nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), # 1024x45x152
                nn.Conv2d(1024, 512, kernel_size=1, stride=1), # 512x45x152
                nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), # 1024x45x152
                nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), # 1024x45x152
                nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1), # 1024x23x76
                
                nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), # 1024x23x76
                nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1), # 1024x11x38
                nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1), # 1024x6x19
            )
        
            # for KITTI object detection, out should be [batch_size, num_objects_Detected, 5]
            self.fully_connected = nn.Sequential(
                nn.Flatten(),
                nn.Linear(6*19*1024, 6*19*2*4*5), # 6x19 is the grid placed on the image
                # nn.Linear(4560, 6*19*2*4*5) # (2 bounding boxes per cell * 4 coordinates and 1 confidence score) + 4 number of classes = 14
            )
            
        # Print the number of training parameters in the model
        self.count_params(self.model)        
            
    def forward(self, data):
        '''
        Forward pass of the model. Modify as other models are added
        
        Args:
            data
            
        Returns:
            output of model
        '''
        
        # Custom linear model
        if self.model_type == "linear":
            out = self.model(data.view(data.size(0),-1))
            return out.reshape(self.batch_size,self.linear_output_dim)
        
        # Custom CNN model
        elif self.model_type == "customcnn":
            out = self.model(data)
            out = out.view(out.size(0), -1)
            out = self.fully_connected(out)
            return out
        
        # Custom model for kitti (copied from YOLO paper)
        elif self.model_type == "customcnn_kitti":
            out = self.model(data)
            out = out.view(out.size(0), -1)
            out = self.fully_connected(out)
            return out
        
    def count_params(Self, model):
        # Print the number of training parameters in the model
        num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"This model has {num_param} parameters")