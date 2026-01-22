"""
Chained Pix2Pix training for PBR map generation.

Implements CHORD-style sequential generation:
  Stage 1: basecolor → normal
  Stage 2: basecolor + normal → roughness
  Stage 3: basecolor + normal + roughness → metallic

Each stage conditions on previous outputs for better coherence.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import yaml
import json

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


class ChainedGeneratorUNet(nn.Module):
    """
    U-Net Generator with variable input channels for chained conditioning.

    - Stage 1 (normal): 3 channels (basecolor)
    - Stage 2 (roughness): 6 channels (basecolor + normal)
    - Stage 3 (metallic): 9 channels (basecolor + normal + roughness)
    """
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.in_channels = in_channels

        # Encoder - first layer adapts to input channels
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


class ChainedDiscriminator(nn.Module):
    """PatchGAN Discriminator with variable input channels."""
    def __init__(self, in_channels=6):
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

    def forward(self, *inputs):
        x = torch.cat(inputs, dim=1)
        return self.model(x)


class ChainedPBRDataset(Dataset):
    """Dataset for chained PBR training with all maps."""
    def __init__(self, data_dir, resolution=512):
        self.data_dir = Path(data_dir) / "chained"
        self.resolution = resolution

        # Check all map directories exist
        self.basecolor_dir = self.data_dir / "basecolor"
        self.normal_dir = self.data_dir / "normal"
        self.roughness_dir = self.data_dir / "roughness"
        self.metallic_dir = self.data_dir / "metallic"

        self.files = sorted(list(self.basecolor_dir.glob("*.png")))
        print(f"Found {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        tensor = torch.from_numpy(np.array(img).astype(np.float32) / 127.5 - 1.0)
        return tensor.permute(2, 0, 1)

    def __getitem__(self, idx):
        filename = self.files[idx].name

        return {
            "basecolor": self._load_image(self.basecolor_dir / filename),
            "normal": self._load_image(self.normal_dir / filename),
            "roughness": self._load_image(self.roughness_dir / filename),
            "metallic": self._load_image(self.metallic_dir / filename),
        }


# Stage configurations
STAGES = {
    "normal": {
        "input_channels": 3,      # basecolor only
        "disc_channels": 6,       # input (3) + target (3)
        "inputs": ["basecolor"],
        "target": "normal",
    },
    "roughness": {
        "input_channels": 6,      # basecolor + normal
        "disc_channels": 9,       # inputs (6) + target (3)
        "inputs": ["basecolor", "normal"],
        "target": "roughness",
    },
    "metallic": {
        "input_channels": 9,      # basecolor + normal + roughness
        "disc_channels": 12,      # inputs (9) + target (3)
        "inputs": ["basecolor", "normal", "roughness"],
        "target": "metallic",
    },
}


def train_stage(config, stage_name: str, prev_generators: dict = None):
    """Train a single stage of the chain."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training Stage: {stage_name.upper()}")
    print(f"{'='*60}")

    stage_config = STAGES[stage_name]
    print(f"Input channels: {stage_config['input_channels']}")
    print(f"Inputs: {stage_config['inputs']}")
    print(f"Target: {stage_config['target']}")

    # Initialize wandb
    if config["logging"]["use_wandb"] and HAS_WANDB:
        wandb.init(
            project=config["logging"]["project_name"],
            config=config,
            name=f"chained-{stage_name}",
            reinit=True,
        )

    # Models
    generator = ChainedGeneratorUNet(
        in_channels=stage_config["input_channels"],
        out_channels=3
    ).to(device)

    discriminator = ChainedDiscriminator(
        in_channels=stage_config["disc_channels"]
    ).to(device)

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
    dataset = ChainedPBRDataset(
        config["training"]["data_dir"],
        config["training"]["resolution"]
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Load previous stage generators for inference (frozen)
    frozen_generators = {}
    if prev_generators:
        for name, gen_path in prev_generators.items():
            stage_cfg = STAGES[name]
            gen = ChainedGeneratorUNet(
                in_channels=stage_cfg["input_channels"],
                out_channels=3
            ).to(device)
            gen.load_state_dict(torch.load(gen_path, map_location=device))
            gen.eval()
            for param in gen.parameters():
                param.requires_grad = False
            frozen_generators[name] = gen
            print(f"Loaded frozen generator: {name}")

    # Training
    epochs = config["training"]["epochs"]
    use_generated = config["training"].get("use_generated_conditioning", False)

    print(f"\nStarting training for {epochs} epochs...")
    print(f"Use generated conditioning: {use_generated}")

    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        epoch_g_loss = 0
        epoch_d_loss = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress:
            # Move all maps to device
            basecolor = batch["basecolor"].to(device)
            normal_gt = batch["normal"].to(device)
            roughness_gt = batch["roughness"].to(device)
            metallic_gt = batch["metallic"].to(device)

            # Build conditioning input based on stage
            if stage_name == "normal":
                cond_input = basecolor
                target = normal_gt

            elif stage_name == "roughness":
                if use_generated and "normal" in frozen_generators:
                    # Use generated normal from previous stage
                    with torch.no_grad():
                        normal_gen = frozen_generators["normal"](basecolor)
                    cond_input = torch.cat([basecolor, normal_gen], dim=1)
                else:
                    # Use ground truth normal
                    cond_input = torch.cat([basecolor, normal_gt], dim=1)
                target = roughness_gt

            elif stage_name == "metallic":
                if use_generated:
                    with torch.no_grad():
                        if "normal" in frozen_generators:
                            normal_gen = frozen_generators["normal"](basecolor)
                        else:
                            normal_gen = normal_gt

                        if "roughness" in frozen_generators:
                            rough_input = torch.cat([basecolor, normal_gen], dim=1)
                            roughness_gen = frozen_generators["roughness"](rough_input)
                        else:
                            roughness_gen = roughness_gt

                    cond_input = torch.cat([basecolor, normal_gen, roughness_gen], dim=1)
                else:
                    cond_input = torch.cat([basecolor, normal_gt, roughness_gt], dim=1)
                target = metallic_gt

            # -----------------
            # Train Generator
            # -----------------
            optimizer_G.zero_grad()

            gen_output = generator(cond_input)

            # GAN loss
            pred_fake = discriminator(cond_input, gen_output)
            valid = torch.ones_like(pred_fake)
            fake = torch.zeros_like(pred_fake)
            loss_GAN = criterion_GAN(pred_fake, valid)

            # L1 loss
            loss_L1 = criterion_L1(gen_output, target)

            loss_G = loss_GAN + lambda_L1 * loss_L1
            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            pred_real = discriminator(cond_input, target)
            loss_real = criterion_GAN(pred_real, valid)

            pred_fake = discriminator(cond_input, gen_output.detach())
            loss_fake = criterion_GAN(pred_fake, fake)

            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()

            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()

            progress.set_postfix(G=f"{loss_G.item():.4f}", D=f"{loss_D.item():.4f}")

        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)

        if config["logging"]["use_wandb"] and HAS_WANDB:
            wandb.log({
                "loss/generator": avg_g_loss,
                "loss/discriminator": avg_d_loss,
                "epoch": epoch + 1,
            })

        print(f"Epoch {epoch+1} - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")

        # Validation
        if (epoch + 1) % config["training"]["validation_epochs"] == 0:
            validate_stage(generator, frozen_generators, dataset, config, stage_name, epoch + 1, device)

        # Save checkpoint
        if (epoch + 1) % config["checkpointing"]["save_epochs"] == 0:
            save_checkpoint(generator, discriminator, config, stage_name, epoch + 1)

    # Save final model
    final_path = save_final(generator, config, stage_name)

    if config["logging"]["use_wandb"] and HAS_WANDB:
        wandb.finish()

    return final_path


def validate_stage(generator, frozen_generators, dataset, config, stage_name, epoch, device):
    """Generate validation images for a stage."""
    import random
    generator.eval()

    print(f"\nValidation at epoch {epoch} ({stage_name})...")

    output_dir = Path(config["checkpointing"]["output_dir"]) / "validation" / stage_name / f"epoch_{epoch}"
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    num_samples = min(4, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    use_generated = config["training"].get("use_generated_conditioning", False)
    all_rows = []

    with torch.no_grad():
        for i in indices:
            sample = dataset[i]
            basecolor = sample["basecolor"].unsqueeze(0).to(device)
            normal_gt = sample["normal"].unsqueeze(0).to(device)
            roughness_gt = sample["roughness"].unsqueeze(0).to(device)

            # Build input
            if stage_name == "normal":
                cond_input = basecolor
                target = sample["normal"]
            elif stage_name == "roughness":
                if use_generated and "normal" in frozen_generators:
                    normal_gen = frozen_generators["normal"](basecolor)
                    cond_input = torch.cat([basecolor, normal_gen], dim=1)
                else:
                    cond_input = torch.cat([basecolor, normal_gt], dim=1)
                target = sample["roughness"]
            elif stage_name == "metallic":
                if use_generated:
                    normal_gen = frozen_generators.get("normal", lambda x: normal_gt)(basecolor)
                    if "roughness" in frozen_generators:
                        rough_input = torch.cat([basecolor, normal_gen], dim=1)
                        roughness_gen = frozen_generators["roughness"](rough_input)
                    else:
                        roughness_gen = roughness_gt
                    cond_input = torch.cat([basecolor, normal_gen, roughness_gen], dim=1)
                else:
                    cond_input = torch.cat([basecolor, normal_gt, roughness_gt], dim=1)
                target = sample["metallic"]

            output = generator(cond_input)

            # Convert to numpy
            basecolor_img = ((sample["basecolor"].permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
            output_img = ((output[0].cpu().permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
            target_img = ((target.permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)

            row = np.concatenate([basecolor_img, output_img, target_img], axis=1)
            all_rows.append(row)

    grid = np.concatenate(all_rows, axis=0)
    grid_img = Image.fromarray(grid)
    grid_img.save(output_dir / "comparison_grid.png")

    if config["logging"]["use_wandb"] and HAS_WANDB:
        wandb.log({
            f"val/{stage_name}_grid": wandb.Image(grid_img, caption=f"Epoch {epoch}"),
        })

    print(f"Saved validation grid to {output_dir / 'comparison_grid.png'}")
    generator.train()


def save_checkpoint(generator, discriminator, config, stage_name, epoch):
    """Save training checkpoint."""
    output_dir = Path(config["checkpointing"]["output_dir"]) / f"chained-{stage_name}-epoch{epoch}"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(generator.state_dict(), output_dir / "generator.pth")
    torch.save(discriminator.state_dict(), output_dir / "discriminator.pth")
    print(f"Saved checkpoint to {output_dir}")


def save_final(generator, config, stage_name):
    """Save final generator model."""
    output_dir = Path(config["checkpointing"]["output_dir"]) / f"chained-{stage_name}-final"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "generator.pth"
    torch.save(generator.state_dict(), model_path)
    print(f"Saved final {stage_name} model to {output_dir}")

    # Save stage config for inference
    stage_config = STAGES[stage_name]
    with open(output_dir / "stage_config.json", "w") as f:
        json.dump(stage_config, f, indent=2)

    return str(model_path)


def train_all_stages(config):
    """Train all stages sequentially."""
    print("\n" + "="*60)
    print("CHAINED PBR TRAINING PIPELINE")
    print("="*60)
    print("\nChain order: basecolor → normal → roughness → metallic")
    print("Each stage conditions on previous outputs.\n")

    prev_generators = {}

    # Stage 1: Normal
    print("\n[Stage 1/3] Training Normal Generator")
    normal_path = train_stage(config, "normal")
    prev_generators["normal"] = normal_path

    # Stage 2: Roughness (conditioned on normal)
    print("\n[Stage 2/3] Training Roughness Generator")
    roughness_path = train_stage(config, "roughness", prev_generators)
    prev_generators["roughness"] = roughness_path

    # Stage 3: Metallic (conditioned on normal + roughness)
    print("\n[Stage 3/3] Training Metallic Generator")
    metallic_path = train_stage(config, "metallic", prev_generators)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nFinal models:")
    print(f"  Normal:    {normal_path}")
    print(f"  Roughness: {roughness_path}")
    print(f"  Metallic:  {metallic_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_pix2pix_chained.yaml")
    parser.add_argument("--stage", type=str, choices=["normal", "roughness", "metallic", "all"],
                        default="all", help="Train specific stage or all stages")
    parser.add_argument("--prev-normal", type=str, help="Path to pretrained normal generator")
    parser.add_argument("--prev-roughness", type=str, help="Path to pretrained roughness generator")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.stage == "all":
        train_all_stages(config)
    else:
        prev_generators = {}
        if args.prev_normal:
            prev_generators["normal"] = args.prev_normal
        if args.prev_roughness:
            prev_generators["roughness"] = args.prev_roughness

        train_stage(config, args.stage, prev_generators if prev_generators else None)


if __name__ == "__main__":
    main()
