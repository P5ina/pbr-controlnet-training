"""
ComfyUI custom nodes for Chained PBR Pix2Pix models.

CHORD-style sequential generation:
  basecolor → normal → roughness → metallic

Each stage conditions on previous outputs for better coherence.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import json
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


class ChainedGeneratorUNet(nn.Module):
    """
    U-Net Generator with variable input channels for chained conditioning.
    """
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.in_channels = in_channels

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

MODELS_DIR = os.path.join(folder_paths.models_dir, "pix2pix_chained")
os.makedirs(MODELS_DIR, exist_ok=True)

# Cache for loaded models
_model_cache = {}

# Stage configurations
STAGES = {
    "normal": {"in_channels": 3},
    "roughness": {"in_channels": 6},
    "metallic": {"in_channels": 9},
}


def get_chained_model_list(stage=None):
    """Get list of available chained models, optionally filtered by stage."""
    if not os.path.exists(MODELS_DIR):
        return []
    models = []
    for f in os.listdir(MODELS_DIR):
        if f.endswith(".pth"):
            if stage is None or stage in f.lower():
                models.append(f)
    return models if models else ["none"]


def load_chained_model(model_name, in_channels, device):
    """Load a chained pix2pix model with caching."""
    cache_key = f"{model_name}_{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    model_path = os.path.join(MODELS_DIR, model_name)
    model = ChainedGeneratorUNet(in_channels=in_channels, out_channels=3)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    _model_cache[cache_key] = model
    return model


def preprocess_image(image, device, target_size=512):
    """Convert ComfyUI image to model input tensor."""
    # ComfyUI: [B, H, W, C] range [0, 1]
    # Model: [B, C, H, W] range [-1, 1]
    img = image[0]  # Take first batch item
    h, w = img.shape[:2]

    if h != target_size or w != target_size:
        img_pil = Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8))
        img_pil = img_pil.resize((target_size, target_size), Image.LANCZOS)
        img = torch.from_numpy(np.array(img_pil).astype(np.float32) / 255.0)

    img_tensor = (img * 2 - 1).permute(2, 0, 1).unsqueeze(0).to(device)
    return img_tensor, (h, w)


def postprocess_output(output, original_size):
    """Convert model output back to ComfyUI format."""
    # Model: [B, C, H, W] range [-1, 1]
    # ComfyUI: [B, H, W, C] range [0, 1]
    output = (output[0].permute(1, 2, 0).cpu() + 1) / 2
    output = output.clamp(0, 1)

    h, w = original_size
    if h != 512 or w != 512:
        out_pil = Image.fromarray((output.numpy() * 255).astype(np.uint8))
        out_pil = out_pil.resize((w, h), Image.LANCZOS)
        output = torch.from_numpy(np.array(out_pil).astype(np.float32) / 255.0)

    return output.unsqueeze(0)


class LoadChainedNormalModel:
    """Load the Normal generator (Stage 1: basecolor → normal)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (get_chained_model_list("normal"),),
            }
        }

    RETURN_TYPES = ("CHAINED_NORMAL_MODEL",)
    RETURN_NAMES = ("normal_model",)
    FUNCTION = "load"
    CATEGORY = "PBR/Chained"

    def load(self, model_name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_chained_model(model_name, in_channels=3, device=device)
        return (model,)


class LoadChainedRoughnessModel:
    """Load the Roughness generator (Stage 2: basecolor + normal → roughness)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (get_chained_model_list("roughness"),),
            }
        }

    RETURN_TYPES = ("CHAINED_ROUGHNESS_MODEL",)
    RETURN_NAMES = ("roughness_model",)
    FUNCTION = "load"
    CATEGORY = "PBR/Chained"

    def load(self, model_name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_chained_model(model_name, in_channels=6, device=device)
        return (model,)


class LoadChainedMetallicModel:
    """Load the Metallic generator (Stage 3: basecolor + normal + roughness → metallic)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (get_chained_model_list("metallic"),),
            }
        }

    RETURN_TYPES = ("CHAINED_METALLIC_MODEL",)
    RETURN_NAMES = ("metallic_model",)
    FUNCTION = "load"
    CATEGORY = "PBR/Chained"

    def load(self, model_name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_chained_model(model_name, in_channels=9, device=device)
        return (model,)


class ChainedNormalInference:
    """Generate normal map from basecolor (Stage 1)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal_model": ("CHAINED_NORMAL_MODEL",),
                "basecolor": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal",)
    FUNCTION = "process"
    CATEGORY = "PBR/Chained"

    def process(self, normal_model, basecolor):
        device = next(normal_model.parameters()).device

        # Preprocess
        bc_tensor, orig_size = preprocess_image(basecolor, device)

        # Inference
        with torch.no_grad():
            normal = normal_model(bc_tensor)

        # Postprocess
        return (postprocess_output(normal, orig_size),)


class ChainedRoughnessInference:
    """Generate roughness map from basecolor + normal (Stage 2)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "roughness_model": ("CHAINED_ROUGHNESS_MODEL",),
                "basecolor": ("IMAGE",),
                "normal": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("roughness",)
    FUNCTION = "process"
    CATEGORY = "PBR/Chained"

    def process(self, roughness_model, basecolor, normal):
        device = next(roughness_model.parameters()).device

        bc_tensor, orig_size = preprocess_image(basecolor, device)
        normal_tensor, _ = preprocess_image(normal, device)

        # Concatenate: [B, 6, H, W]
        input_tensor = torch.cat([bc_tensor, normal_tensor], dim=1)

        with torch.no_grad():
            roughness = roughness_model(input_tensor)

        return (postprocess_output(roughness, orig_size),)


class ChainedMetallicInference:
    """Generate metallic map from basecolor + normal + roughness (Stage 3)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metallic_model": ("CHAINED_METALLIC_MODEL",),
                "basecolor": ("IMAGE",),
                "normal": ("IMAGE",),
                "roughness": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("metallic",)
    FUNCTION = "process"
    CATEGORY = "PBR/Chained"

    def process(self, metallic_model, basecolor, normal, roughness):
        device = next(metallic_model.parameters()).device

        bc_tensor, orig_size = preprocess_image(basecolor, device)
        normal_tensor, _ = preprocess_image(normal, device)
        roughness_tensor, _ = preprocess_image(roughness, device)

        # Concatenate: [B, 9, H, W]
        input_tensor = torch.cat([bc_tensor, normal_tensor, roughness_tensor], dim=1)

        with torch.no_grad():
            metallic = metallic_model(input_tensor)

        return (postprocess_output(metallic, orig_size),)


class ChainedPBRComplete:
    """
    Complete chained PBR generation in one node.

    Pipeline: basecolor → normal → roughness → metallic

    Each stage uses outputs from previous stages for conditioning.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "basecolor": ("IMAGE",),
                "normal_model": ("CHAINED_NORMAL_MODEL",),
                "roughness_model": ("CHAINED_ROUGHNESS_MODEL",),
                "metallic_model": ("CHAINED_METALLIC_MODEL",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("basecolor", "normal", "roughness", "metallic")
    FUNCTION = "process"
    CATEGORY = "PBR/Chained"

    def process(self, basecolor, normal_model, roughness_model, metallic_model):
        device = next(normal_model.parameters()).device

        # Preprocess basecolor
        bc_tensor, orig_size = preprocess_image(basecolor, device)

        with torch.no_grad():
            # Stage 1: basecolor → normal
            normal_tensor = normal_model(bc_tensor)

            # Stage 2: basecolor + normal → roughness
            roughness_input = torch.cat([bc_tensor, normal_tensor], dim=1)
            roughness_tensor = roughness_model(roughness_input)

            # Stage 3: basecolor + normal + roughness → metallic
            metallic_input = torch.cat([bc_tensor, normal_tensor, roughness_tensor], dim=1)
            metallic_tensor = metallic_model(metallic_input)

        # Postprocess all outputs
        normal_out = postprocess_output(normal_tensor, orig_size)
        roughness_out = postprocess_output(roughness_tensor, orig_size)
        metallic_out = postprocess_output(metallic_tensor, orig_size)

        return (basecolor, normal_out, roughness_out, metallic_out)


class ChainedPBRWithExternalNormal:
    """
    Chained PBR generation using an external normal map.

    Useful when you have a better normal map from another source (e.g., DeepBump).
    Pipeline: basecolor + external_normal → roughness → metallic
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "basecolor": ("IMAGE",),
                "external_normal": ("IMAGE",),
                "roughness_model": ("CHAINED_ROUGHNESS_MODEL",),
                "metallic_model": ("CHAINED_METALLIC_MODEL",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("basecolor", "normal", "roughness", "metallic")
    FUNCTION = "process"
    CATEGORY = "PBR/Chained"

    def process(self, basecolor, external_normal, roughness_model, metallic_model):
        device = next(roughness_model.parameters()).device

        bc_tensor, orig_size = preprocess_image(basecolor, device)
        normal_tensor, _ = preprocess_image(external_normal, device)

        with torch.no_grad():
            # Stage 2: basecolor + normal → roughness
            roughness_input = torch.cat([bc_tensor, normal_tensor], dim=1)
            roughness_tensor = roughness_model(roughness_input)

            # Stage 3: basecolor + normal + roughness → metallic
            metallic_input = torch.cat([bc_tensor, normal_tensor, roughness_tensor], dim=1)
            metallic_tensor = metallic_model(metallic_input)

        roughness_out = postprocess_output(roughness_tensor, orig_size)
        metallic_out = postprocess_output(metallic_tensor, orig_size)

        return (basecolor, external_normal, roughness_out, metallic_out)


# ============== Node Mappings ==============

NODE_CLASS_MAPPINGS = {
    # Model loaders
    "LoadChainedNormalModel": LoadChainedNormalModel,
    "LoadChainedRoughnessModel": LoadChainedRoughnessModel,
    "LoadChainedMetallicModel": LoadChainedMetallicModel,
    # Individual stage inference
    "ChainedNormalInference": ChainedNormalInference,
    "ChainedRoughnessInference": ChainedRoughnessInference,
    "ChainedMetallicInference": ChainedMetallicInference,
    # Complete pipelines
    "ChainedPBRComplete": ChainedPBRComplete,
    "ChainedPBRWithExternalNormal": ChainedPBRWithExternalNormal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadChainedNormalModel": "Load Chained Normal Model",
    "LoadChainedRoughnessModel": "Load Chained Roughness Model",
    "LoadChainedMetallicModel": "Load Chained Metallic Model",
    "ChainedNormalInference": "Chained Normal (Stage 1)",
    "ChainedRoughnessInference": "Chained Roughness (Stage 2)",
    "ChainedMetallicInference": "Chained Metallic (Stage 3)",
    "ChainedPBRComplete": "Chained PBR Complete",
    "ChainedPBRWithExternalNormal": "Chained PBR (External Normal)",
}
