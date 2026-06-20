"""TrackNet architecture for ball tracking.

This implements a VGG16-based encoder-decoder architecture (similar to TrackNetV2),
which takes 3 consecutive frames as input (9 channels: 3 frames * 3 RGB) and outputs
a Gaussian heatmap representing the ball's location at time t.
"""

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """A convolutional block matching yastrebksv/TrackNet structure."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)


class TrackNet(nn.Module):
    """TrackNet architecture matching yastrebksv implementation."""

    def __init__(self, in_channels: int = 9, out_channels: int = 256):
        super().__init__()
        
        # Encoder
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)
        
        # Decoder (No skip connections in this specific PyTorch port)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(512, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)
        
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(256, 128)
        self.conv15 = ConvBlock(128, 128)
        
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(128, 64)
        self.conv17 = ConvBlock(64, 64)
        
        # Output layer
        self.conv18 = ConvBlock(64, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        
        # Decoder
        x = self.upsample1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        
        x = self.upsample2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        
        x = self.upsample3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        
        return x
