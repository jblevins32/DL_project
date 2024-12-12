import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            SiLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale


class SPP(nn.Module):
    """Spatial Pyramid Pooling as used in YOLOv3-SPP."""

    def __init__(self):
        super(SPP, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

    def forward(self, x):
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        return torch.cat([x, x1, x2, x3], dim=1)


def conv_block(in_channels, out_channels, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_channels),
        SiLU()
    )


def extra_conv_block(in_channels, out_channels):
    # Just a 3x3 conv block without downsampling
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        SiLU()
    )


class AttentionYOLO(nn.Module):
    def __init__(self, num_classes=4, num_anchors=2):
        super(AttentionYOLO, self).__init__()
        out_channels = num_anchors * (5 + num_classes)

        # Stem
        self.layer1 = conv_block(3, 32, stride=2)  # Downsample: 3->32
        self.layer2 = ResidualBlock(32, 64, stride=2)  # Downsample: 32->64
        self.layer3 = ResidualBlock(64, 128, stride=2)  # Downsample: 64->128

        # Add SE block to enhance channel attention here
        self.se1 = SEBlock(128)

        # An extra residual block instead of extra_conv_block
        self.layer4 = ResidualBlock(128, 128, stride=1)

        self.layer5 = ResidualBlock(128, 256, stride=2)  # 128->256
        self.se2 = SEBlock(256)
        self.layer6 = ResidualBlock(256, 256, stride=1)

        self.layer7 = ResidualBlock(256, 512, stride=2)  # 256->512
        self.se3 = SEBlock(512)
        self.layer8 = ResidualBlock(512, 512, stride=1)

        # Instead of the final_conv_block, we directly downsample to 6x19 if needed
        # Or you can keep a final downsampling block as-is for original dimensions
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(512),
            SiLU()
        )

        # SPP before final prediction: Increases receptive field
        self.spp = SPP()
        # After SPP, channels: 512 * 4 = 2048 because SPP concatenates features
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(512 * 4, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            SiLU()
        )

        # Final Prediction Layer
        self.pred = nn.Conv2d(512, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.layer1(x)  # 32-ch
        x = self.layer2(x)  # 64-ch
        x = self.layer3(x)  # 128-ch
        x = self.se1(x)
        x = self.layer4(x)  # 128-ch
        x = self.layer5(x)  # 256-ch
        x = self.se2(x)
        x = self.layer6(x)  # 256-ch
        x = self.layer7(x)  # 512-ch
        x = self.se3(x)
        x = self.layer8(x)  # 512-ch
        x = self.layer9(x)  # 512-ch (downsample to final size)

        # Apply SPP
        x = self.spp(x)  # 512*4 = 2048-ch
        x = self.reduce_conv(x)  # Back to 512-ch

        x = self.pred(x)
        return x

    def count_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total_params}")
