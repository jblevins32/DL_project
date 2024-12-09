import torch
import torch.nn as nn


class SimpleYOLO(nn.Module):
    def __init__(self, num_classes=5, num_anchors=2):
        super(SimpleYOLO, self).__init__() # initializes subclass functions and variables within nn.Module

        # YOLO-like prediction layer:
        out_channels = num_anchors * (5 + num_classes)

        # Conv block with kernel size 3x3, stride 2 for downsampling
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Additional convolution block without downsampling
        def extra_conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Final block with kernel_size=2, stride=2 to get exactly 6x19
        def final_conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Downsample and increase depth
        # Input: 3 x 365 x 1220
        self.features = nn.Sequential(
            conv_block(3, 32),          # 32 x 182 x 610
            conv_block(32, 64),         # 64 x 91 x 305
            conv_block(64, 128),        # 128 x 46 x 153
            extra_conv_block(128, 128), # 128 x 46 x 153 (extra block)
            conv_block(128, 256),       # 256 x 23 x 77
            extra_conv_block(256, 256), # 256 x 23 x 77 (extra block)
            conv_block(256, 512),       # 512 x 12 x 39
            extra_conv_block(512, 512), # 512 x 12 x 39 (extra block)
            final_conv_block(512, 512), # 512 x 6 x 19
        )

        # A simple 1x1 conv for predictions
        self.pred = nn.Conv2d(512, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.pred(x)
        return x

    def count_params(self):
        # Count the number of trainable parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total_params}")
