"""
ComfyUI custom nodes for PBR Pix2Pix models.
Converts basecolor textures to roughness/metallic maps.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import folder_paths


# ============== Model Architecture ==============

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.model(x)
        return torch.cat([x, skip], dim=1)


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


# ============== ComfyUI Nodes ==============

# Register custom model path
MODELS_DIR = os.path.join(folder_paths.models_dir, "pix2pix")
os.makedirs(MODELS_DIR, exist_ok=True)

# Cache for loaded models
_model_cache = {}


def get_model_list():
    """Get list of available pix2pix models."""
    if not os.path.exists(MODELS_DIR):
        return []
    models = []
    for f in os.listdir(MODELS_DIR):
        if f.endswith(".pth"):
            models.append(f)
    return models


def load_model(model_name, device):
    """Load a pix2pix model with caching."""
    cache_key = f"{model_name}_{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    model_path = os.path.join(MODELS_DIR, model_name)
    model = GeneratorUNet(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    _model_cache[cache_key] = model
    return model


class LoadPix2PixModel:
    """Load a Pix2Pix model for PBR map generation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (get_model_list(),),
            }
        }

    RETURN_TYPES = ("PIX2PIX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "PBR"

    def load(self, model_name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(model_name, device)
        return (model,)


class Pix2PixInference:
    """Apply Pix2Pix model to convert basecolor to roughness/metallic."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PIX2PIX_MODEL",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = "PBR"

    def process(self, model, image):
        device = next(model.parameters()).device

        # ComfyUI images are [B, H, W, C] in range [0, 1]
        # Convert to [B, C, H, W] in range [-1, 1]
        batch_size = image.shape[0]
        results = []

        for i in range(batch_size):
            img = image[i]  # [H, W, C]

            # Resize to 512x512 if needed
            h, w = img.shape[:2]
            if h != 512 or w != 512:
                img_pil = Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8))
                img_pil = img_pil.resize((512, 512), Image.LANCZOS)
                img = torch.from_numpy(np.array(img_pil).astype(np.float32) / 255.0)

            # To tensor [-1, 1], [C, H, W]
            img_tensor = (img * 2 - 1).permute(2, 0, 1).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                output = model(img_tensor)

            # Back to [0, 1], [H, W, C]
            output = (output[0].permute(1, 2, 0).cpu() + 1) / 2
            output = output.clamp(0, 1)

            # Resize back if needed
            if h != 512 or w != 512:
                out_pil = Image.fromarray((output.numpy() * 255).astype(np.uint8))
                out_pil = out_pil.resize((w, h), Image.LANCZOS)
                output = torch.from_numpy(np.array(out_pil).astype(np.float32) / 255.0)

            results.append(output)

        return (torch.stack(results),)


class Pix2PixBatchPBR:
    """Generate all PBR maps from basecolor in one node."""

    @classmethod
    def INPUT_TYPES(cls):
        models = get_model_list()
        return {
            "required": {
                "basecolor": ("IMAGE",),
                "roughness_model": (models,),
                "metallic_model": (models,),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("basecolor", "roughness", "metallic")
    FUNCTION = "process"
    CATEGORY = "PBR"

    def process(self, basecolor, roughness_model, metallic_model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        rough_model = load_model(roughness_model, device)
        metal_model = load_model(metallic_model, device)

        # Process
        roughness_node = Pix2PixInference()
        metallic_node = Pix2PixInference()

        roughness = roughness_node.process(rough_model, basecolor)[0]
        metallic = metallic_node.process(metal_model, basecolor)[0]

        return (basecolor, roughness, metallic)


# ============== Node Mappings ==============

NODE_CLASS_MAPPINGS = {
    "LoadPix2PixModel": LoadPix2PixModel,
    "Pix2PixInference": Pix2PixInference,
    "Pix2PixBatchPBR": Pix2PixBatchPBR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadPix2PixModel": "Load Pix2Pix Model",
    "Pix2PixInference": "Pix2Pix Inference",
    "Pix2PixBatchPBR": "Pix2Pix Batch PBR",
}
