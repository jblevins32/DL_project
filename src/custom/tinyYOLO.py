import torch
import torch.nn as nn


class TinyYOLO(nn.Module):
    def __init__(self, num_classes=5, num_anchors=2):
        super(TinyYOLO, self).__init__()  # initializes subclass functions and variables within nn.Module

        # YOLO-like prediction layer:
        out_channels = num_anchors * (5 + num_classes)

        # Conv block with kernel size 3x3, stride 2 for downsampling
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True)
            )

        # Final block to adjust depth and output size
        def final_conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True)
            )

        # Downsample and increase depth
        # Input: 3 x 365 x 1220
        self.features = nn.Sequential(
            conv_block(3, 16),          # 16 x 182 x 610
            conv_block(16, 32),         # 32 x 91 x 305
            conv_block(32, 64),         # 64 x 46 x 153
            conv_block(64, 128),        # 128 x 23 x 77
            conv_block(128, 256),       # 256 x 12 x 39
        )

        # Final prediction layer
        self.pred = nn.Sequential(
            final_conv_block(256, out_channels),
            nn.AdaptiveAvgPool2d((6, 19))  # Ensure output size is 6x19
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pred(x)
        x = x.view(x.size(0), 1, 18, 6, 19)  # Reshape to 1 x 18 x 6 x 19
        return x

    def count_params(self):
        # Count the number of trainable parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total_params}")
