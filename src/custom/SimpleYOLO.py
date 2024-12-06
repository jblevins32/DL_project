import torch
import torch.nn as nn


class SimpleYOLO(nn.Module):
    def __init__(self, num_classes=4, num_anchors=2):
        super(SimpleYOLO, self).__init__()

        # Initial conv_block function for most layers
        def conv_block(in_channels, out_channels):
            # A simple block with stride=2 conv for downsampling (3x3 kernel)
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Final block with kernel_size=2, stride=2, padding=0 to get exactly 6x19
        def final_conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Downsample aggressively
        # Input: 3 x 365 x 1220
        # After first conv_block:  32 x ~182 x 610
        # After second conv_block: 64 x ~91 x 305
        # After third conv_block:  128 x ~46 x 153
        # After fourth conv_block: 256 x ~23 x 77
        # After fifth conv_block:  512 x ~12 x 39
        # After sixth final_conv_block: 512 x 6 x 19 (exact as needed)

        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            final_conv_block(512, 512),  # Modified last layer
        )

        # YOLO-like prediction layer:
        # Each cell predicts multiple anchors. Each anchor predicts:
        # (x, y, w, h, conf) + class_probs = 5 + num_classes
        # For num_anchors anchors, total channels = num_anchors * (5 + num_classes)
        out_channels = num_anchors * (5 + num_classes)

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
