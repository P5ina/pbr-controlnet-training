"""
ComfyUI custom node for Multi-task PBR generation.

Loads the trained multi-task model and generates all PBR maps from a basecolor input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Try to import ComfyUI modules
try:
    import folder_paths
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False


# ============================================================================
# MODEL DEFINITION (same as training)
# ============================================================================

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, num_heads=8, batch_first=True)
        self.ln = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)
        x_ln = self.ln(x)
        attn_out, _ = self.mha(x_ln, x_ln, x_ln)
        x = x + attn_out
        x = x + self.ff(x)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class OutputHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.conv(x)


class MultiTaskPBRGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision import models

        # EfficientNet-B4 encoder
        efficientnet = models.efficientnet_b4(weights=None)

        self.encoder_stages = nn.ModuleList([
            nn.Sequential(efficientnet.features[0], efficientnet.features[1]),
            efficientnet.features[2],
            efficientnet.features[3],
            efficientnet.features[4],
            nn.Sequential(efficientnet.features[5], efficientnet.features[6],
                         efficientnet.features[7], efficientnet.features[8]),
        ])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(448, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SelfAttention(512),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.decoder4 = DecoderBlock(512, 112, 256)
        self.decoder3 = DecoderBlock(256, 56, 128)
        self.decoder2 = DecoderBlock(128, 32, 64)
        self.decoder1 = DecoderBlock(64, 48, 64)
        self.decoder0 = DecoderBlock(64, 0, 64)

        self.head_normal = OutputHead(64, 3)
        self.head_roughness = OutputHead(64, 1)
        self.head_metallic = OutputHead(64, 1)
        self.head_height = OutputHead(64, 1)

    def forward(self, x):
        skips = []
        for stage in self.encoder_stages:
            x = stage(x)
            skips.append(x)

        x = self.bottleneck(x)
        x = self.decoder4(x, skips[3])
        x = self.decoder3(x, skips[2])
        x = self.decoder2(x, skips[1])
        x = self.decoder1(x, skips[0])
        x = self.decoder0(x)

        return {
            'normal': self.head_normal(x),
            'roughness': self.head_roughness(x),
            'metallic': self.head_metallic(x),
            'height': self.head_height(x),
        }


# ============================================================================
# COMFYUI NODES
# ============================================================================

class LoadMultiTaskPBRModel:
    """Load the multi-task PBR generator model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "models/pbr_multitask.pth"}),
            }
        }

    RETURN_TYPES = ("PBR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "PBR"

    def load_model(self, model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = MultiTaskPBRGenerator()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        return ({"model": model, "device": device},)


class MultiTaskPBRInference:
    """Generate all PBR maps from a basecolor image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PBR_MODEL",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("normal", "roughness", "metallic", "height")
    FUNCTION = "generate"
    CATEGORY = "PBR"

    def generate(self, model, image):
        device = model["device"]
        net = model["model"]

        # Convert ComfyUI image format (B, H, W, C) [0,1] to PyTorch (B, C, H, W) [-1,1]
        x = image.permute(0, 3, 1, 2).to(device)
        x = x * 2 - 1  # [0,1] -> [-1,1]

        # Ensure dimensions are divisible by 32 (for encoder downsampling)
        _, _, h, w = x.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            outputs = net(x)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            for key in outputs:
                outputs[key] = outputs[key][:, :, :h, :w]

        def to_comfy_image(t, expand_grayscale=True):
            """Convert PyTorch tensor to ComfyUI image format."""
            t = (t + 1) / 2  # [-1,1] -> [0,1]
            t = t.clamp(0, 1)
            if t.shape[1] == 1 and expand_grayscale:
                t = t.repeat(1, 3, 1, 1)
            return t.permute(0, 2, 3, 1).cpu()

        normal = to_comfy_image(outputs['normal'], expand_grayscale=False)
        roughness = to_comfy_image(outputs['roughness'])
        metallic = to_comfy_image(outputs['metallic'])
        height = to_comfy_image(outputs['height'])

        return (normal, roughness, metallic, height)


class SplitPBRChannels:
    """Split a 3-channel normal map into individual channels for further processing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal_map": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("normal_x", "normal_y", "normal_z")
    FUNCTION = "split"
    CATEGORY = "PBR"

    def split(self, normal_map):
        # Normal map is (B, H, W, 3)
        x = normal_map[:, :, :, 0:1].repeat(1, 1, 1, 3)
        y = normal_map[:, :, :, 1:2].repeat(1, 1, 1, 3)
        z = normal_map[:, :, :, 2:3].repeat(1, 1, 1, 3)
        return (x, y, z)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "LoadMultiTaskPBRModel": LoadMultiTaskPBRModel,
    "MultiTaskPBRInference": MultiTaskPBRInference,
    "SplitPBRChannels": SplitPBRChannels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadMultiTaskPBRModel": "Load Multi-Task PBR Model",
    "MultiTaskPBRInference": "Multi-Task PBR Inference",
    "SplitPBRChannels": "Split PBR Normal Channels",
}
