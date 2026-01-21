"""
Pix2Pix training for PBR map generation.
Trains a U-Net generator to translate basecolor → roughness/metallic.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import yaml

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class UNetDown(nn.Module):
    """Downsampling block for U-Net."""
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
    """Upsampling block for U-Net."""
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
    """U-Net Generator for Pix2Pix."""
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # Encoder
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # Decoder
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
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


class Discriminator(nn.Module):
    """PatchGAN Discriminator."""
    def __init__(self, in_channels=6):  # input + target concatenated
        super().__init__()

        def block(in_ch, out_ch, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, img_input, img_target):
        x = torch.cat([img_input, img_target], dim=1)
        return self.model(x)


class PBRDataset(Dataset):
    """Dataset for PBR map training."""
    def __init__(self, data_dir, target_map, resolution=512):
        self.data_dir = Path(data_dir) / target_map
        self.cond_dir = self.data_dir / "conditioning"
        self.target_dir = self.data_dir / "target"
        self.resolution = resolution

        self.files = sorted(list(self.cond_dir.glob("*.png")))
        print(f"Found {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        cond_path = self.files[idx]
        target_path = self.target_dir / cond_path.name

        # Load images
        cond_img = Image.open(cond_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")

        # Resize
        cond_img = cond_img.resize((self.resolution, self.resolution), Image.LANCZOS)
        target_img = target_img.resize((self.resolution, self.resolution), Image.LANCZOS)

        # To tensor [-1, 1]
        cond = torch.from_numpy(np.array(cond_img).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        target = torch.from_numpy(np.array(target_img).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)

        return {"conditioning": cond, "target": target}


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb
    if config["logging"]["use_wandb"] and HAS_WANDB:
        wandb.init(
            project=config["logging"]["project_name"],
            config=config,
            name=f"pix2pix-{config['training']['target_map']}",
        )

    # Models
    generator = GeneratorUNet(in_channels=3, out_channels=3).to(device)
    discriminator = Discriminator(in_channels=6).to(device)

    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator params: {g_params:,}")
    print(f"Discriminator params: {d_params:,}")

    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(),
        lr=config["training"]["learning_rate"],
        betas=(0.5, 0.999)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(),
        lr=config["training"]["learning_rate"],
        betas=(0.5, 0.999)
    )

    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_L1 = nn.L1Loss()
    lambda_L1 = config["training"]["lambda_l1"]

    # Dataset
    dataset = PBRDataset(
        config["training"]["data_dir"],
        config["training"]["target_map"],
        config["training"]["resolution"]
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Training
    epochs = config["training"]["epochs"]
    global_step = 0

    print(f"\nStarting training for {epochs} epochs...")
    print(f"Target map: {config['training']['target_map']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Lambda L1: {lambda_L1}")

    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        epoch_g_loss = 0
        epoch_d_loss = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress:
            cond = batch["conditioning"].to(device)
            target = batch["target"].to(device)

            # -----------------
            # Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate
            gen_output = generator(cond)

            # GAN loss
            pred_fake = discriminator(cond, gen_output)
            valid = torch.ones_like(pred_fake)
            fake = torch.zeros_like(pred_fake)
            loss_GAN = criterion_GAN(pred_fake, valid)

            # L1 loss
            loss_L1 = criterion_L1(gen_output, target)

            # Total generator loss
            loss_G = loss_GAN + lambda_L1 * loss_L1

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(cond, target)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(cond, gen_output.detach())
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total discriminator loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()

            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()
            global_step += 1

            progress.set_postfix(G=f"{loss_G.item():.4f}", D=f"{loss_D.item():.4f}", L1=f"{loss_L1.item():.4f}")

            # Log to wandb
            if config["logging"]["use_wandb"] and HAS_WANDB and global_step % 50 == 0:
                wandb.log({
                    "loss/generator": loss_G.item(),
                    "loss/discriminator": loss_D.item(),
                    "loss/l1": loss_L1.item(),
                    "loss/gan": loss_GAN.item(),
                    "step": global_step,
                })

        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        print(f"Epoch {epoch+1} - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")

        # Validation
        if (epoch + 1) % config["training"]["validation_epochs"] == 0:
            validate(generator, dataset, config, epoch + 1, device)

        # Save checkpoint
        if (epoch + 1) % config["checkpointing"]["save_epochs"] == 0:
            save_checkpoint(generator, discriminator, config, epoch + 1)

    # Save final model
    save_final(generator, config)

    if config["logging"]["use_wandb"] and HAS_WANDB:
        wandb.finish()


def validate(generator, dataset, config, epoch, device):
    """Generate validation images."""
    generator.eval()

    target_map = config["training"]["target_map"]
    print(f"\nValidation at epoch {epoch} ({target_map})...")

    output_dir = Path(config["checkpointing"]["output_dir"]) / "validation" / f"epoch_{epoch}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Take first 4 samples
    all_rows = []

    with torch.no_grad():
        for i in range(min(4, len(dataset))):
            sample = dataset[i]
            cond = sample["conditioning"].unsqueeze(0).to(device)
            target = sample["target"]

            output = generator(cond)

            # Convert to numpy images [0, 255]
            cond_img = ((sample["conditioning"].permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
            output_img = ((output[0].cpu().permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
            target_img = ((target.permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)

            # Create row: input | output | target
            row = np.concatenate([cond_img, output_img, target_img], axis=1)
            all_rows.append(row)

    # Stack all rows
    grid = np.concatenate(all_rows, axis=0)
    grid_img = Image.fromarray(grid)
    grid_img.save(output_dir / "comparison_grid.png")

    # Log to wandb
    if config["logging"]["use_wandb"] and HAS_WANDB:
        wandb.log({
            "val/comparison_grid": wandb.Image(grid_img, caption=f"Epoch {epoch}"),
            "epoch": epoch,
        })

    print(f"Saved validation grid to {output_dir / 'comparison_grid.png'}")
    generator.train()


def save_checkpoint(generator, discriminator, config, epoch):
    """Save training checkpoint."""
    target_map = config["training"]["target_map"]
    output_dir = Path(config["checkpointing"]["output_dir"]) / f"pix2pix-{target_map}-epoch{epoch}"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(generator.state_dict(), output_dir / "generator.pth")
    torch.save(discriminator.state_dict(), output_dir / "discriminator.pth")
    print(f"Saved checkpoint to {output_dir}")


def save_final(generator, config):
    """Save final generator model."""
    target_map = config["training"]["target_map"]
    output_dir = Path(config["checkpointing"]["output_dir"]) / f"pix2pix-{target_map}-final"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(generator.state_dict(), output_dir / "generator.pth")
    print(f"Saved final model to {output_dir}")
    print(f"  - generator.pth (PyTorch)")

    # Try to save as ONNX for easier deployment
    try:
        dummy_input = torch.randn(1, 3, 512, 512)
        generator.cpu().eval()
        torch.onnx.export(
            generator,
            dummy_input,
            output_dir / "generator.onnx",
            input_names=["basecolor"],
            output_names=["output"],
            dynamic_axes={"basecolor": {0: "batch"}, "output": {0: "batch"}},
            opset_version=11
        )
        print(f"  - generator.onnx (ONNX)")
    except Exception as e:
        print(f"  - ONNX export skipped: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_pix2pix.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
