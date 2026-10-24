"""EfficientNet-B4 / U-Net++ with CBAM before the output head."""

import torch
from torch import nn
import segmentation_models_pytorch as smp

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class CBAM_UnetPlusPlus(nn.Module):
    def __init__(self, encoder_name, encoder_weights, in_channels, classes, activation):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )
        decoder_channels = self.model.decoder.out_channels[-1] if hasattr(self.model.decoder, "out_channels") else 256
        self.cbam = CBAM(decoder_channels)
    def forward(self, x):
        features = self.model.encoder(x)
        decoder_outputs = self.model.decoder(features)
        out = decoder_outputs[-1] if isinstance(decoder_outputs, (list, tuple)) else decoder_outputs
        if out.dim() == 3:
            out = out.unsqueeze(0)
        elif out.dim() != 4:
            raise RuntimeError(f"CBAM expects [B,C,H,W], got {out.shape}")
        out = self.cbam(out)
        masks = self.model.segmentation_head(out)
        return masks

def build_model():
    """Construct the seven-channel network without downloading weights."""
    return CBAM_UnetPlusPlus("efficientnet-b4", None, 7, 1, "sigmoid")
