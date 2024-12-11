import torch.nn as nn

class EncoderDecoderYOLO(nn.Module):
    def __init__(self, num_classes=5, num_anchors=2):
        super(EncoderDecoderYOLO, self).__init__()

        # Output channels for the final prediction
        out_channels = num_anchors * (5 + num_classes)

        # Encoder block
        def conv_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True)
            )

        # Decoder block
        def upsample_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True)
            )

        # Encoder: Downsampling and increasing depth
        self.encoder = nn.Sequential(
            conv_block(3, 32),          # 32 x 182 x 610
            conv_block(32, 64),         # 64 x 91 x 305
            conv_block(64, 128),        # 128 x 46 x 153
            conv_block(128, 256),       # 256 x 23 x 77
            conv_block(256, 512),       # 512 x 12 x 39
        )

        # Bottleneck: Reduce spatial size further
        self.bottleneck = conv_block(512, 1024, kernel_size=2, stride=2, padding=0)  # 1024 x 6 x 19

        # Decoder: Upsampling and reducing depth
        self.decoder = nn.Sequential(
            upsample_block(1024, 512),  # 512 x 12 x 39
            upsample_block(512, 256),   # 256 x 24 x 78
            upsample_block(256, 128),   # 128 x 48 x 156
            upsample_block(128, 64),    # 64 x 96 x 312
            upsample_block(64, 32),     # 32 x 192 x 624
        )

        # Final 1x1 conv to get desired output size
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)  # Adjust depth to num_anchors * (5 + num_classes)

        # Output reshape layer
        self.output_reshape = nn.Conv2d(out_channels, 18, kernel_size=1)  # Final depth adjustment

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        x = self.bottleneck(x)

        # Decoder
        x = self.decoder(x)

        # Final convolution to adjust output depth
        x = self.final_conv(x)

        # Reshape output to (batch_size, 18, 6, 19)
        x = self.output_reshape(x)
        x = nn.functional.interpolate(x, size=(6, 19), mode='bilinear', align_corners=False)  # Ensure 6x9 spatial dimensions
        return x