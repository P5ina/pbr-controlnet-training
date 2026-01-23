"""
ComfyUI custom node for Multi-task PBR generation with text conditioning.

Loads the trained multi-task model and generates all PBR maps from a basecolor input.
Supports optional text prompts for better material understanding.
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

# Try to import CLIP for text conditioning
try:
    from transformers import CLIPTextModel, CLIPTokenizer
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False


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

    def forward(self, x, text_emb=None):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)
        x_ln = self.ln(x)
        attn_out, _ = self.mha(x_ln, x_ln, x_ln)
        x = x + attn_out
        x = x + self.ff(x)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x


class CrossAttention(nn.Module):
    """Cross-attention for text conditioning."""
    def __init__(self, channels, context_dim=512):
        super().__init__()
        self.channels = channels
        self.context_dim = context_dim

        self.ln_q = nn.LayerNorm(channels)
        self.ln_kv = nn.LayerNorm(context_dim)

        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(context_dim, channels)
        self.to_v = nn.Linear(context_dim, channels)
        self.to_out = nn.Linear(channels, channels)

        self.num_heads = 8
        self.head_dim = channels // self.num_heads

        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x, context):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)

        x_ln = self.ln_q(x)
        context_ln = self.ln_kv(context)

        q = self.to_q(x_ln)
        k = self.to_k(context_ln)
        v = self.to_v(context_ln)

        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, -1, C)
        out = self.to_out(out)

        x = x + out
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
    """Model without text conditioning (for backward compatibility)."""
    def __init__(self):
        super().__init__()
        from torchvision import models

        efficientnet = models.efficientnet_b4(weights=None)

        self.encoder_stages = nn.ModuleList([
            nn.Sequential(efficientnet.features[0], efficientnet.features[1]),
            efficientnet.features[2],
            efficientnet.features[3],
            efficientnet.features[4],
            nn.Sequential(efficientnet.features[5], efficientnet.features[6],
                         efficientnet.features[7]),  # Exclude features[8] (1792ch head)
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
        self.decoder1 = DecoderBlock(64, 24, 64)  # 24ch from first encoder stage
        self.decoder0 = DecoderBlock(64, 0, 64)

        self.head_normal = OutputHead(64, 3)
        self.head_roughness = OutputHead(64, 1)
        self.head_metallic = OutputHead(64, 1)
        self.head_height = OutputHead(64, 1)

    def forward(self, x, text_emb=None):
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


class MultiTaskPBRGeneratorWithText(nn.Module):
    """Model with text conditioning."""
    def __init__(self):
        super().__init__()
        from torchvision import models

        efficientnet = models.efficientnet_b4(weights=None)

        self.encoder_stages = nn.ModuleList([
            nn.Sequential(efficientnet.features[0], efficientnet.features[1]),
            efficientnet.features[2],
            efficientnet.features[3],
            efficientnet.features[4],
            nn.Sequential(efficientnet.features[5], efficientnet.features[6],
                         efficientnet.features[7]),  # Exclude features[8] (1792ch head)
        ])

        # Bottleneck with self-attention and cross-attention
        self.bottleneck_conv1 = nn.Sequential(
            nn.Conv2d(448, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.bottleneck_self_attn = SelfAttention(512)
        self.bottleneck_cross_attn = CrossAttention(512, context_dim=512)
        self.bottleneck_conv2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.decoder4 = DecoderBlock(512, 112, 256)
        self.decoder3 = DecoderBlock(256, 56, 128)
        self.decoder2 = DecoderBlock(128, 32, 64)
        self.decoder1 = DecoderBlock(64, 24, 64)  # 24ch from first encoder stage
        self.decoder0 = DecoderBlock(64, 0, 64)

        self.head_normal = OutputHead(64, 3)
        self.head_roughness = OutputHead(64, 1)
        self.head_metallic = OutputHead(64, 1)
        self.head_height = OutputHead(64, 1)

    def forward(self, x, text_emb=None):
        skips = []
        for stage in self.encoder_stages:
            x = stage(x)
            skips.append(x)

        x = self.bottleneck_conv1(x)
        x = self.bottleneck_self_attn(x)

        if text_emb is not None:
            x = self.bottleneck_cross_attn(x, text_emb)

        x = self.bottleneck_conv2(x)

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
                "use_text_conditioning": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("PBR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "PBR"

    def load_model(self, model_path, use_text_conditioning):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Choose model architecture
        if use_text_conditioning:
            model = MultiTaskPBRGeneratorWithText()
        else:
            model = MultiTaskPBRGenerator()

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        # Load CLIP text encoder if using text conditioning
        text_encoder = None
        tokenizer = None
        if use_text_conditioning and HAS_CLIP:
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            text_encoder.to(device)
            text_encoder.eval()

        return ({
            "model": model,
            "device": device,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "use_text": use_text_conditioning and HAS_CLIP,
        },)


class MultiTaskPBRInference:
    """Generate all PBR maps from a basecolor image with optional text prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PBR_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "e.g., metal texture, rusty iron, weathered"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("normal", "roughness", "metallic", "height")
    FUNCTION = "generate"
    CATEGORY = "PBR"

    def generate(self, model, image, prompt=""):
        device = model["device"]
        net = model["model"]
        text_encoder = model.get("text_encoder")
        tokenizer = model.get("tokenizer")
        use_text = model.get("use_text", False)

        # Convert ComfyUI image format (B, H, W, C) [0,1] to PyTorch (B, C, H, W) [-1,1]
        x = image.permute(0, 3, 1, 2).to(device)
        x = x * 2 - 1  # [0,1] -> [-1,1]

        # Ensure dimensions are divisible by 32 (for encoder downsampling)
        _, _, h, w = x.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # Encode text prompt
        text_emb = None
        if use_text and text_encoder is not None and tokenizer is not None and prompt.strip():
            with torch.no_grad():
                inputs = tokenizer(
                    [prompt],
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt"
                ).to(device)
                text_emb = text_encoder(**inputs).last_hidden_state

        with torch.no_grad():
            outputs = net(x, text_emb=text_emb)

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
