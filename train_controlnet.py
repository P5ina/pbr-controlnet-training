"""
Train ControlNet for PBR map generation.

Trains a ControlNet to convert basecolor images to roughness/metallic maps.

Usage:
    accelerate launch train_controlnet.py --config config.yaml
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import yaml

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class PBRControlNetDataset(Dataset):
    """Dataset for ControlNet training with basecolor→target pairs."""

    def __init__(
        self,
        data_dir: str,
        target_map: str = "roughness",
        resolution: int = 512,
    ):
        self.data_dir = Path(data_dir) / target_map
        self.resolution = resolution
        self.target_map = target_map

        self.cond_dir = self.data_dir / "conditioning"
        self.target_dir = self.data_dir / "target"

        # Load prompts
        prompts_file = self.data_dir / "prompts.json"
        with open(prompts_file) as f:
            self.prompts = json.load(f)

        self.filenames = list(self.prompts.keys())
        print(f"Loaded {len(self.filenames)} samples for {target_map}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # Load conditioning image (basecolor)
        cond_path = self.cond_dir / filename
        cond_img = Image.open(cond_path).convert("RGB")
        cond_img = cond_img.resize((self.resolution, self.resolution), Image.LANCZOS)

        # Load target image (roughness/metallic)
        target_path = self.target_dir / filename
        target_img = Image.open(target_path).convert("RGB")
        target_img = target_img.resize((self.resolution, self.resolution), Image.LANCZOS)

        # Convert to tensors [-1, 1]
        cond_tensor = torch.from_numpy(
            np.array(cond_img).astype(np.float32) / 127.5 - 1.0
        ).permute(2, 0, 1)

        target_tensor = torch.from_numpy(
            np.array(target_img).astype(np.float32) / 127.5 - 1.0
        ).permute(2, 0, 1)

        return {
            "conditioning": cond_tensor,
            "target": target_tensor,
            "prompt": self.prompts[filename],
        }


def train(config: dict):
    """Main training loop."""
    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        mixed_precision=config["training"]["mixed_precision"],
        log_with="wandb" if config["logging"]["use_wandb"] and HAS_WANDB else None,
    )

    set_seed(config["training"].get("seed", 42))

    # Initialize wandb
    if accelerator.is_main_process and config["logging"]["use_wandb"] and HAS_WANDB:
        wandb.init(
            project=config["logging"]["project_name"],
            config=config,
            name=f"controlnet-{config['training']['target_map']}",
        )

    # Load models
    print("Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(
        config["model"]["base_model"], subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config["model"]["base_model"], subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        config["model"]["base_model"], subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        config["model"]["base_model"], subfolder="unet"
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        config["model"]["base_model"], subfolder="scheduler"
    )

    # Create ControlNet from UNet
    print("Creating ControlNet...")
    controlnet = ControlNetModel.from_unet(unet)

    # Freeze everything except ControlNet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Enable gradient checkpointing
    if config["training"]["gradient_checkpointing"]:
        controlnet.enable_gradient_checkpointing()

    # Enable xformers for memory efficiency
    try:
        import xformers
        controlnet.enable_xformers_memory_efficient_attention()
        unet.enable_xformers_memory_efficient_attention()
        print("xformers enabled")
    except ImportError:
        print("xformers not available, using default attention")

    # Dataset
    print("Loading dataset...")
    dataset = PBRControlNetDataset(
        config["training"]["data_dir"],
        config["training"]["target_map"],
        config["training"]["resolution"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=config["training"]["learning_rate"],
        betas=(config["training"]["adam_beta1"], config["training"]["adam_beta2"]),
        weight_decay=config["training"]["adam_weight_decay"],
    )

    # LR scheduler
    lr_scheduler = get_scheduler(
        config["training"]["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["training"]["lr_warmup_steps"],
        num_training_steps=config["training"]["max_train_steps"],
    )

    # Prepare with accelerator
    controlnet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, dataloader, lr_scheduler
    )

    # Move frozen models to device with fp16 to save memory
    weight_dtype = torch.float16 if config["training"]["mixed_precision"] == "fp16" else torch.float32
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Enable VAE slicing for lower memory
    vae.enable_slicing()

    # Training loop
    global_step = 0
    max_steps = config["training"]["max_train_steps"]

    print(f"\nStarting training for {max_steps} steps...")
    print(f"Target map: {config['training']['target_map']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Gradient accumulation: {config['training']['gradient_accumulation_steps']}")

    progress = tqdm(total=max_steps, desc="Training")

    while global_step < max_steps:
        controlnet.train()

        for batch in dataloader:
            with accelerator.accumulate(controlnet):
                # Encode conditioning image (basecolor)
                conditioning = batch["conditioning"].to(accelerator.device, dtype=weight_dtype)

                # Encode target image
                target = batch["target"].to(accelerator.device, dtype=weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(target).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=accelerator.device
                ).long()

                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Encode prompts
                input_ids = tokenizer(
                    batch["prompt"],
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(accelerator.device)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]

                # ControlNet forward
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=conditioning,
                    return_dict=False,
                )

                # UNet forward with ControlNet residuals
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                # Loss
                loss = F.mse_loss(noise_pred, noise, reduction="mean")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        controlnet.parameters(),
                        config["training"]["max_grad_norm"]
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)

                # Logging
                if global_step % config["logging"]["log_every_n_steps"] == 0:
                    log_dict = {
                        "loss": loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                    }

                    if config["logging"]["use_wandb"] and HAS_WANDB and accelerator.is_main_process:
                        wandb.log(log_dict)

                    progress.set_postfix(loss=f"{loss.item():.4f}")

                # Validation
                if global_step % config["training"]["validation_steps"] == 0:
                    if accelerator.is_main_process:
                        validate_and_log(
                            controlnet, unet, vae, text_encoder, tokenizer,
                            noise_scheduler, config, global_step, accelerator.device,
                            weight_dtype=weight_dtype
                        )

                # Save checkpoint
                if global_step % config["checkpointing"]["save_steps"] == 0:
                    if accelerator.is_main_process:
                        save_checkpoint(controlnet, config, global_step)

                if global_step >= max_steps:
                    break

    progress.close()

    # Final save
    if accelerator.is_main_process:
        save_checkpoint(controlnet, config, global_step, final=True)

    if config["logging"]["use_wandb"] and HAS_WANDB:
        wandb.finish()

    print("Training complete!")


@torch.no_grad()
def validate_and_log(
    controlnet, unet, vae, text_encoder, tokenizer,
    scheduler, config, step, device, weight_dtype=torch.float16
):
    """Generate validation images and create comparison grid."""
    controlnet.eval()

    target_map = config["training"]["target_map"]
    print(f"\nValidation at step {step} ({target_map})...")

    # Load sample conditioning images
    data_dir = Path(config["training"]["data_dir"]) / target_map
    cond_dir = data_dir / "conditioning"
    target_dir = data_dir / "target"

    sample_files = list(cond_dir.glob("*.png"))[:4]

    if not sample_files:
        print("No validation images found")
        return

    all_inputs = []
    all_outputs = []
    all_targets = []

    for sample_file in sample_files:
        # Load conditioning (basecolor)
        cond_img = Image.open(sample_file).convert("RGB")
        cond_img = cond_img.resize((512, 512), Image.LANCZOS)
        cond_tensor = torch.from_numpy(
            np.array(cond_img).astype(np.float32) / 127.5 - 1.0
        ).permute(2, 0, 1).unsqueeze(0).to(device, dtype=weight_dtype)

        # Load ground truth target
        target_file = target_dir / sample_file.name
        if target_file.exists():
            target_img = Image.open(target_file).convert("RGB")
            target_img = target_img.resize((512, 512), Image.LANCZOS)
        else:
            target_img = Image.new("RGB", (512, 512), (128, 128, 128))

        # Encode prompt
        prompt = f"pbr {target_map} map, high quality, seamless texture"
        input_ids = tokenizer(
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        ).input_ids.to(device)
        encoder_hidden_states = text_encoder(input_ids)[0]

        # Generate with more steps for better quality
        latents = torch.randn(1, 4, 64, 64, device=device, dtype=weight_dtype)
        scheduler.set_timesteps(30, device=device)

        for t in scheduler.timesteps:
            down_samples, mid_sample = controlnet(
                latents, t, encoder_hidden_states,
                controlnet_cond=cond_tensor, return_dict=False
            )
            noise_pred = unet(
                latents, t, encoder_hidden_states,
                down_block_additional_residuals=down_samples,
                mid_block_additional_residual=mid_sample,
            ).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Decode
        latents = latents / vae.config.scaling_factor
        image = vae.decode(latents).sample
        image = (image.clamp(-1, 1) + 1) / 2
        output_img = (image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        all_inputs.append(np.array(cond_img))
        all_outputs.append(output_img)
        all_targets.append(np.array(target_img))

    # Create comparison grid: Input | Output | Ground Truth
    grid_rows = []
    for inp, out, tgt in zip(all_inputs, all_outputs, all_targets):
        row = np.concatenate([inp, out, tgt], axis=1)
        grid_rows.append(row)

    grid = np.concatenate(grid_rows, axis=0)

    # Save locally
    out_dir = Path(config["checkpointing"]["output_dir"]) / "validation" / f"step_{step}"
    out_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(grid).save(out_dir / "comparison_grid.png")

    for i, (inp, out, tgt) in enumerate(zip(all_inputs, all_outputs, all_targets)):
        Image.fromarray(inp).save(out_dir / f"{i:02d}_input_basecolor.png")
        Image.fromarray(out).save(out_dir / f"{i:02d}_output_{target_map}.png")
        Image.fromarray(tgt).save(out_dir / f"{i:02d}_target_{target_map}.png")

    # Log to wandb
    if config["logging"]["use_wandb"] and HAS_WANDB:
        # Log comparison grid
        wandb.log({
            f"val/comparison_grid": wandb.Image(
                grid,
                caption=f"Step {step} | Input (basecolor) | Output ({target_map}) | Ground Truth"
            ),
        }, step=step)

        # Log individual samples
        for i, (inp, out, tgt) in enumerate(zip(all_inputs, all_outputs, all_targets)):
            wandb.log({
                f"val/sample_{i}/input": wandb.Image(inp, caption="Basecolor"),
                f"val/sample_{i}/output": wandb.Image(out, caption=f"Generated {target_map}"),
                f"val/sample_{i}/target": wandb.Image(tgt, caption=f"Ground truth {target_map}"),
            }, step=step)

    print(f"Validation images saved to {out_dir}")
    print(f"  - comparison_grid.png (Input | Output | Ground Truth)")
    controlnet.train()


def save_checkpoint(controlnet, config, step, final=False):
    """Save ControlNet checkpoint."""
    out_dir = Path(config["checkpointing"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    target_map = config["training"]["target_map"]

    if final:
        save_path = out_dir / f"controlnet-{target_map}-final"
    else:
        save_path = out_dir / f"controlnet-{target_map}-{step}"

    # Unwrap if needed
    model = controlnet.module if hasattr(controlnet, "module") else controlnet
    model.save_pretrained(save_path)

    print(f"Checkpoint saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
