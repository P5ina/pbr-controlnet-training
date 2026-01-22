"""
Multi-task PBR Generator with Pretrained Encoder and Attention.

Architecture:
  - Pretrained EfficientNet-B4 encoder (frozen initially, then fine-tuned)
  - Self-attention in bottleneck for global context
  - Shared decoder trunk with skip connections
  - Separate heads for: Normal (3ch), Roughness (1ch), Metallic (1ch), Height (1ch)

Losses:
  - L1 reconstruction loss
  - VGG perceptual loss
  - SSIM structural loss
  - Gradient loss for edge preservation
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import yaml
import math

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ============================================================================
# LOSSES
# ============================================================================

class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss."""
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        # Use features from multiple layers
        self.blocks = nn.ModuleList([
            vgg[:4].eval(),   # relu1_2
            vgg[4:9].eval(),  # relu2_2
            vgg[9:16].eval(), # relu3_3
        ])

        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        self.to(device)

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        # Input is [-1, 1], convert to [0, 1] then normalize
        x = (x + 1) / 2
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        # For single-channel outputs, repeat to 3 channels
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        pred = self.normalize(pred)
        target = self.normalize(target)

        loss = 0
        for block in self.blocks:
            pred = block(pred)
            target = block(target)
            loss += F.l1_loss(pred, target)

        return loss


class SSIMLoss(nn.Module):
    """Structural Similarity Loss."""
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def gaussian_window(self, size, sigma, channels):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.outer(g).unsqueeze(0).unsqueeze(0)
        return window.repeat(channels, 1, 1, 1)

    def forward(self, pred, target):
        channels = pred.shape[1]
        window = self.gaussian_window(self.window_size, 1.5, channels).to(pred.device)

        mu_pred = F.conv2d(pred, window, padding=self.window_size//2, groups=channels)
        mu_target = F.conv2d(target, window, padding=self.window_size//2, groups=channels)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred ** 2, window, padding=self.window_size//2, groups=channels) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, window, padding=self.window_size//2, groups=channels) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, window, padding=self.window_size//2, groups=channels) - mu_pred_target

        ssim = ((2 * mu_pred_target + self.C1) * (2 * sigma_pred_target + self.C2)) / \
               ((mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2))

        return 1 - ssim.mean()


class GradientLoss(nn.Module):
    """Gradient/Edge preservation loss using Sobel operators."""
    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, pred, target):
        # Process each channel separately
        loss = 0
        for c in range(pred.shape[1]):
            pred_c = pred[:, c:c+1]
            target_c = target[:, c:c+1]

            pred_gx = F.conv2d(pred_c, self.sobel_x, padding=1)
            pred_gy = F.conv2d(pred_c, self.sobel_y, padding=1)
            target_gx = F.conv2d(target_c, self.sobel_x, padding=1)
            target_gy = F.conv2d(target_c, self.sobel_y, padding=1)

            loss += F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)

        return loss / pred.shape[1]


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class SelfAttention(nn.Module):
    """Self-attention block for global context."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
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
        # Reshape to sequence
        x = x.view(B, C, H * W).permute(0, 2, 1)  # B, HW, C

        # Self-attention
        x_ln = self.ln(x)
        attn_out, _ = self.mha(x_ln, x_ln, x_ln)
        x = x + attn_out

        # Feed-forward
        x = x + self.ff(x)

        # Reshape back
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x


class DecoderBlock(nn.Module):
    """Decoder block with skip connection."""
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
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class OutputHead(nn.Module):
    """Task-specific output head."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, x):
        return self.conv(x)


class MultiTaskPBRGenerator(nn.Module):
    """
    Multi-task PBR map generator.

    Input: Basecolor image (3 channels)
    Output: Normal (3ch), Roughness (1ch), Metallic (1ch), Height (1ch)
    """
    def __init__(self, pretrained=True, freeze_encoder_epochs=5):
        super().__init__()
        self.freeze_encoder_epochs = freeze_encoder_epochs

        # Pretrained EfficientNet-B4 encoder
        efficientnet = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Extract encoder stages
        self.encoder_stages = nn.ModuleList([
            nn.Sequential(efficientnet.features[0], efficientnet.features[1]),  # 48ch, /2
            efficientnet.features[2],   # 32ch, /4
            efficientnet.features[3],   # 56ch, /8
            efficientnet.features[4],   # 112ch, /16
            nn.Sequential(efficientnet.features[5], efficientnet.features[6], efficientnet.features[7], efficientnet.features[8]),  # 448ch, /32
        ])

        # Channel counts from EfficientNet-B4
        self.encoder_channels = [48, 32, 56, 112, 448]

        # Bottleneck with self-attention
        self.bottleneck = nn.Sequential(
            nn.Conv2d(448, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SelfAttention(512),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Shared decoder
        self.decoder4 = DecoderBlock(512, 112, 256)   # /16 -> /8
        self.decoder3 = DecoderBlock(256, 56, 128)    # /8 -> /4
        self.decoder2 = DecoderBlock(128, 32, 64)     # /4 -> /2
        self.decoder1 = DecoderBlock(64, 48, 64)      # /2 -> /1
        self.decoder0 = DecoderBlock(64, 0, 64)       # Final upsample

        # Task-specific heads
        self.head_normal = OutputHead(64, 3)
        self.head_roughness = OutputHead(64, 1)
        self.head_metallic = OutputHead(64, 1)
        self.head_height = OutputHead(64, 1)

    def freeze_encoder(self):
        """Freeze encoder weights."""
        for stage in self.encoder_stages:
            for param in stage.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning."""
        for stage in self.encoder_stages:
            for param in stage.parameters():
                param.requires_grad = True

    def forward(self, x):
        # Encoder
        skips = []
        for stage in self.encoder_stages:
            x = stage(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        x = self.decoder4(x, skips[3])  # Use /16 skip
        x = self.decoder3(x, skips[2])  # Use /8 skip
        x = self.decoder2(x, skips[1])  # Use /4 skip
        x = self.decoder1(x, skips[0])  # Use /2 skip
        x = self.decoder0(x)            # Final

        # Task heads
        normal = self.head_normal(x)
        roughness = self.head_roughness(x)
        metallic = self.head_metallic(x)
        height = self.head_height(x)

        return {
            'normal': normal,
            'roughness': roughness,
            'metallic': metallic,
            'height': height,
        }


# ============================================================================
# DATASET
# ============================================================================

class PBRDataset(Dataset):
    """Dataset for multi-task PBR training."""
    def __init__(self, data_dir, resolution=512, augment=True):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.augment = augment

        # Find all materials
        self.basecolor_dir = self.data_dir / "basecolor"
        self.normal_dir = self.data_dir / "normal"
        self.roughness_dir = self.data_dir / "roughness"
        self.metallic_dir = self.data_dir / "metallic"
        self.height_dir = self.data_dir / "height"

        # Support both jpg and png
        self.files = sorted(
            list(self.basecolor_dir.glob("*.jpg")) +
            list(self.basecolor_dir.glob("*.png"))
        )

        print(f"Found {len(self.files)} materials in {data_dir}")

        # Check which maps are available
        self.has_height = self.height_dir.exists() and any(self.height_dir.iterdir())
        if not self.has_height:
            print("  Note: Height maps not found, will skip height output")

    def __len__(self):
        return len(self.files)

    def _load_image(self, path, grayscale=False):
        """Load and preprocess image."""
        if not path.exists():
            # Return zeros if file doesn't exist
            channels = 1 if grayscale else 3
            return torch.zeros(channels, self.resolution, self.resolution)

        img = Image.open(path)
        if grayscale:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        img = img.resize((self.resolution, self.resolution), Image.LANCZOS)

        # Convert to tensor [-1, 1]
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0

        if grayscale:
            return torch.from_numpy(arr).unsqueeze(0)
        else:
            return torch.from_numpy(arr).permute(2, 0, 1)

    def _find_file(self, directory, base_name):
        """Find file with matching name (jpg or png)."""
        for ext in ['.jpg', '.png', '.jpeg']:
            path = directory / (base_name + ext)
            if path.exists():
                return path
        # Try original extension
        return directory / base_name

    def __getitem__(self, idx):
        base_path = self.files[idx]
        base_name = base_path.stem

        # Load all maps
        basecolor = self._load_image(base_path, grayscale=False)
        normal = self._load_image(self._find_file(self.normal_dir, base_name), grayscale=False)
        roughness = self._load_image(self._find_file(self.roughness_dir, base_name), grayscale=True)
        metallic = self._load_image(self._find_file(self.metallic_dir, base_name), grayscale=True)

        if self.has_height:
            height = self._load_image(self._find_file(self.height_dir, base_name), grayscale=True)
        else:
            height = torch.zeros(1, self.resolution, self.resolution)

        # Augmentation: random horizontal/vertical flips and 90° rotations
        if self.augment:
            # All maps must be augmented the same way
            if torch.rand(1) > 0.5:
                basecolor = torch.flip(basecolor, [2])
                normal = torch.flip(normal, [2])
                normal[0] = -normal[0]  # Flip normal X component
                roughness = torch.flip(roughness, [2])
                metallic = torch.flip(metallic, [2])
                height = torch.flip(height, [2])

            if torch.rand(1) > 0.5:
                basecolor = torch.flip(basecolor, [1])
                normal = torch.flip(normal, [1])
                normal[1] = -normal[1]  # Flip normal Y component
                roughness = torch.flip(roughness, [1])
                metallic = torch.flip(metallic, [1])
                height = torch.flip(height, [1])

            # Random 90° rotation
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                basecolor = torch.rot90(basecolor, k, [1, 2])
                normal = torch.rot90(normal, k, [1, 2])
                # Rotate normal XY components
                if k == 1:
                    normal = torch.stack([-normal[1], normal[0], normal[2]], dim=0)
                elif k == 2:
                    normal = torch.stack([-normal[0], -normal[1], normal[2]], dim=0)
                elif k == 3:
                    normal = torch.stack([normal[1], -normal[0], normal[2]], dim=0)
                roughness = torch.rot90(roughness, k, [1, 2])
                metallic = torch.rot90(metallic, k, [1, 2])
                height = torch.rot90(height, k, [1, 2])

        return {
            'basecolor': basecolor,
            'normal': normal,
            'roughness': roughness,
            'metallic': metallic,
            'height': height,
            'name': base_name,
        }


# ============================================================================
# TRAINING
# ============================================================================

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Initialize wandb
    if config["logging"]["use_wandb"] and HAS_WANDB:
        wandb.init(
            project=config["logging"]["project_name"],
            config=config,
            name="multitask-pbr",
        )

    # Model
    model = MultiTaskPBRGenerator(
        pretrained=config["model"]["pretrained"],
        freeze_encoder_epochs=config["model"]["freeze_encoder_epochs"],
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Freeze encoder initially
    if config["model"]["freeze_encoder_epochs"] > 0:
        model.freeze_encoder()
        print(f"Encoder frozen for first {config['model']['freeze_encoder_epochs']} epochs")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
        eta_min=config["training"]["learning_rate"] / 100,
    )

    # Loss functions
    criterion_l1 = nn.L1Loss()
    criterion_perceptual = VGGPerceptualLoss(device)
    criterion_ssim = SSIMLoss()
    criterion_gradient = GradientLoss().to(device)

    # Loss weights
    lambda_l1 = config["training"]["lambda_l1"]
    lambda_perceptual = config["training"]["lambda_perceptual"]
    lambda_ssim = config["training"]["lambda_ssim"]
    lambda_gradient = config["training"]["lambda_gradient"]

    # Dataset
    dataset = PBRDataset(
        config["training"]["data_dir"],
        config["training"]["resolution"],
        augment=config["training"]["augment"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    # Training loop
    epochs = config["training"]["epochs"]
    global_step = 0
    best_loss = float('inf')

    print(f"\nStarting training for {epochs} epochs...")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Resolution: {config['training']['resolution']}")
    print(f"Loss weights: L1={lambda_l1}, Perceptual={lambda_perceptual}, SSIM={lambda_ssim}, Gradient={lambda_gradient}")

    for epoch in range(epochs):
        model.train()

        # Unfreeze encoder after warmup
        if epoch == config["model"]["freeze_encoder_epochs"]:
            model.unfreeze_encoder()
            print(f"\nEpoch {epoch+1}: Encoder unfrozen for fine-tuning")

        epoch_losses = {
            'total': 0, 'l1': 0, 'perceptual': 0, 'ssim': 0, 'gradient': 0,
            'normal': 0, 'roughness': 0, 'metallic': 0, 'height': 0,
        }

        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress:
            basecolor = batch['basecolor'].to(device)
            normal_gt = batch['normal'].to(device)
            roughness_gt = batch['roughness'].to(device)
            metallic_gt = batch['metallic'].to(device)
            height_gt = batch['height'].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(basecolor)

            # Compute losses for each output
            losses = {}
            total_loss = 0

            for name, pred, gt in [
                ('normal', outputs['normal'], normal_gt),
                ('roughness', outputs['roughness'], roughness_gt),
                ('metallic', outputs['metallic'], metallic_gt),
                ('height', outputs['height'], height_gt),
            ]:
                # Skip height if not available
                if name == 'height' and not dataset.has_height:
                    continue

                l1 = criterion_l1(pred, gt)
                perceptual = criterion_perceptual(pred, gt)
                ssim = criterion_ssim(pred, gt)
                gradient = criterion_gradient(pred, gt)

                task_loss = (
                    lambda_l1 * l1 +
                    lambda_perceptual * perceptual +
                    lambda_ssim * ssim +
                    lambda_gradient * gradient
                )

                losses[name] = task_loss.item()
                total_loss += task_loss

                epoch_losses[name] += task_loss.item()

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            epoch_losses['total'] += total_loss.item()
            global_step += 1

            progress.set_postfix(
                loss=f"{total_loss.item():.4f}",
                N=f"{losses.get('normal', 0):.3f}",
                R=f"{losses.get('roughness', 0):.3f}",
                M=f"{losses.get('metallic', 0):.3f}",
            )

            # Log to wandb
            if config["logging"]["use_wandb"] and HAS_WANDB and global_step % 50 == 0:
                wandb.log({
                    "loss/total": total_loss.item(),
                    "loss/normal": losses.get('normal', 0),
                    "loss/roughness": losses.get('roughness', 0),
                    "loss/metallic": losses.get('metallic', 0),
                    "loss/height": losses.get('height', 0),
                    "lr": optimizer.param_groups[0]['lr'],
                    "step": global_step,
                })

        # Update learning rate
        scheduler.step()

        # Epoch summary
        n_batches = len(dataloader)
        avg_loss = epoch_losses['total'] / n_batches
        print(f"\nEpoch {epoch+1} - Loss: {avg_loss:.4f} | "
              f"N: {epoch_losses['normal']/n_batches:.4f} | "
              f"R: {epoch_losses['roughness']/n_batches:.4f} | "
              f"M: {epoch_losses['metallic']/n_batches:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Validation
        if (epoch + 1) % config["training"]["validation_epochs"] == 0:
            validate(model, dataset, config, epoch + 1, device)

        # Save checkpoint
        if (epoch + 1) % config["checkpointing"]["save_epochs"] == 0:
            save_checkpoint(model, optimizer, scheduler, config, epoch + 1, avg_loss)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_best(model, config, epoch + 1, avg_loss)

    # Save final model
    save_final(model, config)

    if config["logging"]["use_wandb"] and HAS_WANDB:
        wandb.finish()


def validate(model, dataset, config, epoch, device):
    """Generate validation images."""
    import random
    model.eval()

    print(f"\nValidation at epoch {epoch}...")

    output_dir = Path(config["checkpointing"]["output_dir"]) / "validation" / f"epoch_{epoch}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fixed samples for consistent comparison
    random.seed(42)
    num_samples = min(4, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    with torch.no_grad():
        all_rows = []

        for i in indices:
            sample = dataset[i]
            basecolor = sample['basecolor'].unsqueeze(0).to(device)

            outputs = model(basecolor)

            # Convert to numpy images [0, 255]
            def to_img(t, is_grayscale=False):
                t = t.cpu()
                if is_grayscale:
                    t = t.repeat(3, 1, 1)  # Convert to RGB for display
                arr = ((t.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
                return arr

            basecolor_img = to_img(sample['basecolor'])
            normal_pred = to_img(outputs['normal'][0])
            normal_gt = to_img(sample['normal'])
            roughness_pred = to_img(outputs['roughness'][0], is_grayscale=True)
            roughness_gt = to_img(sample['roughness'], is_grayscale=True)
            metallic_pred = to_img(outputs['metallic'][0], is_grayscale=True)
            metallic_gt = to_img(sample['metallic'], is_grayscale=True)

            # Create comparison row: basecolor | normal_pred | normal_gt | rough_pred | rough_gt | metal_pred | metal_gt
            row = np.concatenate([
                basecolor_img, normal_pred, normal_gt,
                roughness_pred, roughness_gt,
                metallic_pred, metallic_gt,
            ], axis=1)
            all_rows.append(row)

        # Stack all rows
        grid = np.concatenate(all_rows, axis=0)
        grid_img = Image.fromarray(grid)
        grid_img.save(output_dir / "comparison_grid.png")

        # Log to wandb
        if config["logging"]["use_wandb"] and HAS_WANDB:
            wandb.log({
                "val/comparison_grid": wandb.Image(grid_img, caption=f"Epoch {epoch} - Base|NormP|NormGT|RoughP|RoughGT|MetalP|MetalGT"),
                "epoch": epoch,
            })

    print(f"Saved validation grid to {output_dir / 'comparison_grid.png'}")
    model.train()


def save_checkpoint(model, optimizer, scheduler, config, epoch, loss):
    """Save training checkpoint."""
    output_dir = Path(config["checkpointing"]["output_dir"]) / f"checkpoint-epoch{epoch}"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, output_dir / "checkpoint.pth")

    print(f"Saved checkpoint to {output_dir}")


def save_best(model, config, epoch, loss):
    """Save best model."""
    output_dir = Path(config["checkpointing"]["output_dir"]) / "best"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / "model.pth")

    # Save info
    with open(output_dir / "info.txt", "w") as f:
        f.write(f"Epoch: {epoch}\nLoss: {loss:.6f}\n")

    print(f"Saved best model (epoch {epoch}, loss {loss:.4f})")


def save_final(model, config):
    """Save final model with ONNX export."""
    output_dir = Path(config["checkpointing"]["output_dir"]) / "final"
    output_dir.mkdir(parents=True, exist_ok=True)

    # PyTorch model
    torch.save(model.state_dict(), output_dir / "model.pth")
    print(f"Saved final model to {output_dir / 'model.pth'}")

    # ONNX export
    try:
        model.eval()
        model.cpu()
        dummy_input = torch.randn(1, 3, 512, 512)

        torch.onnx.export(
            model,
            dummy_input,
            output_dir / "model.onnx",
            input_names=["basecolor"],
            output_names=["normal", "roughness", "metallic", "height"],
            dynamic_axes={
                "basecolor": {0: "batch", 2: "height", 3: "width"},
                "normal": {0: "batch", 2: "height", 3: "width"},
                "roughness": {0: "batch", 2: "height", 3: "width"},
                "metallic": {0: "batch", 2: "height", 3: "width"},
                "height": {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=14,
        )
        print(f"Saved ONNX model to {output_dir / 'model.onnx'}")
    except Exception as e:
        print(f"ONNX export failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_multitask.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
