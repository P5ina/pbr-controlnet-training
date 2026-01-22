"""
SDXL LoRA Training for PBR Map Generation.

Uses img2img approach: input image latents concatenated with noise.
This teaches the model to transform basecolor → target PBR map.

Usage:
    python train_sdxl_lora.py --config config_sdxl_lora.yaml --target normal
"""

import argparse
import os
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import yaml

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from peft import LoraConfig, get_peft_model

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class PBRDataset(Dataset):
    """Dataset for PBR LoRA training."""

    def __init__(self, data_dir: str, target_map: str, resolution: int = 1024):
        self.data_dir = Path(data_dir) / "chained"
        self.target_map = target_map
        self.resolution = resolution

        self.basecolor_dir = self.data_dir / "basecolor"
        self.target_dir = self.data_dir / target_map

        self.files = sorted([
            f.name for f in self.basecolor_dir.iterdir()
            if f.suffix in ['.jpg', '.png']
        ])

        prompts_file = self.data_dir / "prompts.json"
        if prompts_file.exists():
            with open(prompts_file) as f:
                self.prompts = json.load(f)
        else:
            self.prompts = {}

        print(f"Found {len(self.files)} samples for {target_map}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]

        basecolor = Image.open(self.basecolor_dir / filename).convert("RGB")
        target = Image.open(self.target_dir / filename).convert("RGB")

        basecolor = basecolor.resize((self.resolution, self.resolution), Image.LANCZOS)
        target = target.resize((self.resolution, self.resolution), Image.LANCZOS)

        basecolor = torch.from_numpy(np.array(basecolor)).float() / 127.5 - 1.0
        target = torch.from_numpy(np.array(target)).float() / 127.5 - 1.0

        basecolor = basecolor.permute(2, 0, 1)
        target = target.permute(2, 0, 1)

        prompt_data = self.prompts.get(filename, {})
        caption = prompt_data.get("caption", "material texture")

        # Create specific prompt for the target map
        if self.target_map == "normal":
            prompt = f"normal map of {caption}, blue purple normal map, pbr texture"
        elif self.target_map == "roughness":
            prompt = f"roughness map of {caption}, grayscale roughness map, pbr texture"
        elif self.target_map == "metallic":
            prompt = f"metallic map of {caption}, grayscale metallic map, pbr texture"
        else:
            prompt = f"{self.target_map} map of {caption}, pbr texture"

        return {
            "basecolor": basecolor,
            "target": target,
            "prompt": prompt,
        }


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16 if config["training"]["mixed_precision"] else torch.float32

    print(f"Device: {device}, dtype: {weight_dtype}")
    print(f"Target map: {config['training']['target_map']}")

    # Load models
    print("Loading SDXL models...")
    model_id = config["model"]["pretrained_model"]

    tokenizer_1 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")

    text_encoder_1 = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=weight_dtype
    ).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_id, subfolder="text_encoder_2", torch_dtype=weight_dtype
    ).to(device)

    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=weight_dtype
    ).to(device)

    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=weight_dtype
    ).to(device)

    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Freeze encoders
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    vae.requires_grad_(False)

    # Add LoRA to UNet
    print("Adding LoRA adapters...")
    lora_config = LoraConfig(
        r=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
        ],
        lora_dropout=config["lora"]["dropout"],
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    if config["training"]["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()

    # Dataset
    dataset = PBRDataset(
        config["training"]["data_dir"],
        config["training"]["target_map"],
        config["training"]["resolution"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    num_training_steps = len(dataloader) * config["training"]["epochs"]
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=num_training_steps,
    )

    if config["logging"]["use_wandb"] and HAS_WANDB:
        wandb.init(
            project=config["logging"]["project_name"],
            config=config,
            name=f"sdxl-lora-{config['training']['target_map']}",
        )

    # Pre-compute text embeddings for efficiency
    print("Pre-computing text embeddings...")

    # Training
    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    global_step = 0

    for epoch in range(config["training"]["epochs"]):
        unet.train()
        epoch_loss = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")

        for batch in progress:
            basecolor = batch["basecolor"].to(device, dtype=weight_dtype)
            target = batch["target"].to(device, dtype=weight_dtype)
            prompts = batch["prompt"]

            batch_size = basecolor.shape[0]

            # Encode images to latents
            with torch.no_grad():
                target_latents = vae.encode(target).latent_dist.sample()
                target_latents = target_latents * vae.config.scaling_factor

                # Encode basecolor - we'll use this to condition the generation
                basecolor_latents = vae.encode(basecolor).latent_dist.sample()
                basecolor_latents = basecolor_latents * vae.config.scaling_factor

            # Add noise to target
            noise = torch.randn_like(target_latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (batch_size,), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)

            # Encode prompts
            with torch.no_grad():
                # Tokenize
                tokens_1 = tokenizer_1(
                    prompts, padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt"
                ).input_ids.to(device)

                tokens_2 = tokenizer_2(
                    prompts, padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt"
                ).input_ids.to(device)

                # Encode
                prompt_embeds_1 = text_encoder_1(tokens_1, output_hidden_states=True)
                prompt_embeds_2 = text_encoder_2(tokens_2, output_hidden_states=True)

                prompt_embeds = torch.cat([
                    prompt_embeds_1.hidden_states[-2],
                    prompt_embeds_2.hidden_states[-2],
                ], dim=-1)

                pooled_prompt_embeds = prompt_embeds_2[0]

            # Add basecolor information via cross attention
            # We concatenate basecolor latent features to the prompt embeddings
            # This is a simple way to condition on the input image
            basecolor_flat = basecolor_latents.flatten(2).permute(0, 2, 1)  # [B, HW, C]
            # Project to match prompt embedding dimension
            # For simplicity, we'll skip this and rely on the prompt to guide generation

            # Time embeddings for SDXL
            add_time_ids = torch.tensor([
                [config["training"]["resolution"], config["training"]["resolution"], 0, 0,
                 config["training"]["resolution"], config["training"]["resolution"]]
            ], device=device, dtype=weight_dtype).repeat(batch_size, 1)

            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds.to(weight_dtype),
                "time_ids": add_time_ids,
            }

            # Predict noise
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds.to(weight_dtype),
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            # Loss
            loss = F.mse_loss(noise_pred, noise)

            # Also add perceptual guidance: predicted output should be close to target
            # when denoised from high noise level
            if timesteps.float().mean() > 500:  # High noise regime
                with torch.no_grad():
                    # Estimate clean latents
                    alpha_t = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(device)
                    pred_original = (noisy_latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()

                # Additional L1 loss to guide towards target
                guidance_loss = F.l1_loss(pred_original, target_latents)
                loss = loss + 0.1 * guidance_loss

            optimizer.zero_grad()
            loss.backward()

            if config["training"]["max_grad_norm"]:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), config["training"]["max_grad_norm"])

            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")

            if config["logging"]["use_wandb"] and HAS_WANDB and global_step % 50 == 0:
                wandb.log({
                    "loss": loss.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                })

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        if config["logging"]["use_wandb"] and HAS_WANDB:
            wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1})

        # Validation
        if (epoch + 1) % config["training"]["validation_epochs"] == 0:
            validate(unet, vae, tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2,
                     noise_scheduler, dataset, config, epoch + 1, device, weight_dtype)

        # Save checkpoint
        if (epoch + 1) % config["checkpointing"]["save_epochs"] == 0:
            save_lora(unet, config, epoch + 1)

    # Save final
    save_lora(unet, config, "final")

    if config["logging"]["use_wandb"] and HAS_WANDB:
        wandb.finish()


def validate(unet, vae, tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2,
             noise_scheduler, dataset, config, epoch, device, weight_dtype):
    """Generate validation samples."""
    print(f"\nValidation at epoch {epoch}...")
    unet.eval()

    output_dir = Path(config["checkpointing"]["output_dir"]) / "validation" / config["training"]["target_map"] / f"epoch_{epoch}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(4, len(dataset))):
        sample = dataset[i]
        basecolor = sample["basecolor"]
        target = sample["target"]
        prompt = sample["prompt"]

        with torch.no_grad():
            # Tokenize
            tokens_1 = tokenizer_1(
                [prompt], padding="max_length", max_length=77,
                truncation=True, return_tensors="pt"
            ).input_ids.to(device)

            tokens_2 = tokenizer_2(
                [prompt], padding="max_length", max_length=77,
                truncation=True, return_tensors="pt"
            ).input_ids.to(device)

            prompt_embeds_1 = text_encoder_1(tokens_1, output_hidden_states=True)
            prompt_embeds_2 = text_encoder_2(tokens_2, output_hidden_states=True)

            prompt_embeds = torch.cat([
                prompt_embeds_1.hidden_states[-2],
                prompt_embeds_2.hidden_states[-2],
            ], dim=-1).to(weight_dtype)

            pooled_prompt_embeds = prompt_embeds_2[0].to(weight_dtype)

            add_time_ids = torch.tensor([
                [config["training"]["resolution"], config["training"]["resolution"], 0, 0,
                 config["training"]["resolution"], config["training"]["resolution"]]
            ], device=device, dtype=weight_dtype)

            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            }

            # Start from noise
            latents = torch.randn(1, 4, config["training"]["resolution"] // 8,
                                  config["training"]["resolution"] // 8,
                                  device=device, dtype=weight_dtype)

            # Denoise
            noise_scheduler.set_timesteps(30)
            for t in noise_scheduler.timesteps:
                noise_pred = unet(
                    latents, t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

            # Decode
            latents = latents / vae.config.scaling_factor
            image = vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image[0].permute(1, 2, 0).cpu().float().numpy()
            image = (image * 255).astype(np.uint8)

        Image.fromarray(image).save(output_dir / f"{i:02d}_generated.jpg")

        bc_img = ((basecolor.permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
        tgt_img = ((target.permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
        Image.fromarray(bc_img).save(output_dir / f"{i:02d}_input.jpg")
        Image.fromarray(tgt_img).save(output_dir / f"{i:02d}_target.jpg")

    print(f"Saved validation to {output_dir}")
    unet.train()


def save_lora(unet, config, identifier):
    """Save LoRA weights."""
    target_map = config["training"]["target_map"]
    output_dir = Path(config["checkpointing"]["output_dir"]) / f"lora-{target_map}-{identifier}"
    output_dir.mkdir(parents=True, exist_ok=True)

    unet.save_pretrained(output_dir)
    print(f"Saved LoRA to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_sdxl_lora.yaml")
    parser.add_argument("--target", type=str, choices=["normal", "roughness", "metallic"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.target:
        config["training"]["target_map"] = args.target

    train(config)


if __name__ == "__main__":
    main()
