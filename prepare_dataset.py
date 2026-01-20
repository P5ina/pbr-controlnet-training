"""
Prepare MatSynth dataset for ControlNet training.

Creates paired datasets:
- basecolor → roughness
- basecolor → metallic

Usage:
    python prepare_dataset.py --output ./data --max-samples 4000
"""

import os
import argparse
from pathlib import Path
import json

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def process_image(img, target_size: int = 512) -> Image.Image:
    """Process and resize image."""
    if not isinstance(img, Image.Image):
        return None

    if img.width != target_size or img.height != target_size:
        img = img.resize((target_size, target_size), Image.LANCZOS)

    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def prepare_controlnet_dataset(
    output_dir: str,
    target_map: str = "roughness",
    split: str = "train",
    max_samples: int = None,
    resolution: int = 512,
):
    """
    Prepare dataset for ControlNet training.

    Creates structure:
        output_dir/
            conditioning/  (basecolor images)
            target/        (roughness or metallic images)
            prompts.json   (captions)
    """
    output_path = Path(output_dir) / target_map
    cond_path = output_path / "conditioning"
    target_path = output_path / "target"

    cond_path.mkdir(parents=True, exist_ok=True)
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading MatSynth dataset ({split} split)...")
    print(f"Target map: {target_map}")
    print(f"Resolution: {resolution}x{resolution}")

    dataset = load_dataset(
        "gvecchio/MatSynth",
        split=split,
        streaming=True,
    )

    processed = 0
    skipped = 0
    prompts = {}

    total_estimate = max_samples if max_samples else 4000
    print(f"\nProcessing materials (target: {total_estimate})...")

    for idx, sample in enumerate(tqdm(dataset, total=total_estimate, unit="mat")):
        if max_samples and processed >= max_samples:
            break

        try:
            # Get basecolor
            basecolor_img = None
            for key in ["basecolor", "diffuse", "basecolor_map"]:
                if key in sample and sample[key] is not None:
                    basecolor_img = process_image(sample[key], resolution)
                    break

            if basecolor_img is None:
                skipped += 1
                continue

            # Get target map (roughness or metallic)
            target_img = None
            for key in [target_map, f"{target_map}_map"]:
                if key in sample and sample[key] is not None:
                    target_img = process_image(sample[key], resolution)
                    break

            if target_img is None:
                skipped += 1
                continue

            # Generate filename
            filename = f"{processed:05d}.png"

            # Save images
            basecolor_img.save(cond_path / filename, "PNG")
            target_img.save(target_path / filename, "PNG")

            # Get caption
            metadata = sample.get("metadata", {}) or {}
            material_name = str(sample.get("name", f"material_{idx}"))
            category = str(sample.get("category", "unknown"))

            if isinstance(metadata, dict):
                description = metadata.get("description", "")
                tags = metadata.get("tags", [])
            else:
                description = ""
                tags = []

            if description:
                caption = description
            elif tags and isinstance(tags, list):
                caption = ", ".join(tags)
            else:
                caption = f"{material_name}, {category}"

            # Add target-specific suffix
            caption = f"{caption}, {target_map} map"
            prompts[filename] = caption

            processed += 1

        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            skipped += 1
            continue

    # Save prompts
    with open(output_path / "prompts.json", "w") as f:
        json.dump(prompts, f, indent=2)

    # Save metadata
    meta = {
        "target_map": target_map,
        "total_samples": processed,
        "skipped": skipped,
        "resolution": resolution,
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Dataset prepared for: {target_map}")
    print(f"Total pairs: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Output: {output_path}")
    print(f"\nStructure:")
    print(f"  {cond_path}/ - basecolor images (conditioning)")
    print(f"  {target_path}/ - {target_map} images (target)")
    print(f"  {output_path}/prompts.json - captions")


def main():
    parser = argparse.ArgumentParser(description="Prepare MatSynth for ControlNet training")
    parser.add_argument("--output", type=str, default="./data")
    parser.add_argument("--target", type=str, default="roughness",
                        choices=["roughness", "metallic"])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--all", action="store_true",
                        help="Prepare both roughness and metallic datasets")

    args = parser.parse_args()

    if args.all:
        for target in ["roughness", "metallic"]:
            prepare_controlnet_dataset(
                args.output, target, args.split, args.max_samples, args.resolution
            )
    else:
        prepare_controlnet_dataset(
            args.output, args.target, args.split, args.max_samples, args.resolution
        )


if __name__ == "__main__":
    main()
