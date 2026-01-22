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
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Thread
import io

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def is_mostly_black(img: Image.Image, threshold: float = 0.05) -> bool:
    """Check if image is mostly black (mean brightness < threshold)."""
    if img is None:
        return True
    arr = np.array(img.convert("L")).astype(float) / 255.0
    return arr.mean() < threshold


def process_image(img, target_size: int = 512) -> Image.Image:
    """Process and resize image."""
    if not isinstance(img, Image.Image):
        return None

    if img.width != target_size or img.height != target_size:
        img = img.resize((target_size, target_size), Image.LANCZOS)

    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def save_image_fast(img: Image.Image, path: Path):
    """Save image with optimized settings."""
    img.save(path, "PNG", compress_level=1)  # Faster compression


def process_and_save(item, cond_path, target_path, target_map, resolution, filter_empty=False):
    """Process a single sample and save images."""
    idx, sample, processed_idx = item

    try:
        # Get basecolor
        basecolor_img = None
        for key in ["basecolor", "diffuse", "basecolor_map"]:
            if key in sample and sample[key] is not None:
                basecolor_img = process_image(sample[key], resolution)
                break

        if basecolor_img is None:
            return None

        # Get target map (roughness or metallic)
        target_img = None
        raw_target = None
        for key in [target_map, f"{target_map}_map"]:
            if key in sample and sample[key] is not None:
                raw_target = sample[key]
                target_img = process_image(sample[key], resolution)
                break

        if target_img is None:
            return None

        # Filter out mostly black images (for metallic)
        if filter_empty and is_mostly_black(raw_target):
            return None

        # Generate filename
        filename = f"{processed_idx:05d}.png"

        # Save images
        save_image_fast(basecolor_img, cond_path / filename)
        save_image_fast(target_img, target_path / filename)

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

        caption = f"{caption}, {target_map} map"

        return (filename, caption)

    except Exception as e:
        return None


def prepare_controlnet_dataset(
    output_dir: str,
    target_map: str = "roughness",
    split: str = "train",
    max_samples: int = None,
    resolution: int = 512,
    num_workers: int = 8,
    filter_empty: bool = False,
):
    """
    Prepare dataset for ControlNet training with parallel processing.
    """
    output_path = Path(output_dir) / target_map
    cond_path = output_path / "conditioning"
    target_path = output_path / "target"

    cond_path.mkdir(parents=True, exist_ok=True)
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading MatSynth dataset ({split} split)...")
    print(f"Target map: {target_map}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Workers: {num_workers}")
    if filter_empty:
        print(f"Filtering: skipping mostly-black target maps")

    # Download full dataset (faster with good internet)
    dataset = load_dataset(
        "gvecchio/MatSynth",
        split=split,
    )

    processed = 0
    skipped = 0
    prompts = {}

    total_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    print(f"\nProcessing {total_samples} materials...")

    # Collect samples in batches for parallel processing
    batch_size = num_workers * 4
    batch = []

    pbar = tqdm(total=total_samples, unit="mat")

    for idx, sample in enumerate(dataset):
        if max_samples and processed >= max_samples:
            break

        batch.append((idx, sample, processed))
        processed += 1

        # Process batch in parallel
        if len(batch) >= batch_size or (max_samples and processed >= max_samples):
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        process_and_save, item, cond_path, target_path, target_map, resolution, filter_empty
                    ): item for item in batch
                }

                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        filename, caption = result
                        prompts[filename] = caption
                        pbar.update(1)
                    else:
                        skipped += 1
                        pbar.update(1)

            batch = []

    # Process remaining
    if batch:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    process_and_save, item, cond_path, target_path, target_map, resolution
                ): item for item in batch
            }

            for future in as_completed(futures):
                result = future.result()
                if result:
                    filename, caption = result
                    prompts[filename] = caption
                    pbar.update(1)
                else:
                    skipped += 1
                    pbar.update(1)

    pbar.close()

    # Save prompts
    with open(output_path / "prompts.json", "w") as f:
        json.dump(prompts, f, indent=2)

    # Save metadata
    meta = {
        "target_map": target_map,
        "total_samples": len(prompts),
        "skipped": skipped,
        "resolution": resolution,
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Dataset prepared for: {target_map}")
    print(f"Total pairs: {len(prompts)}")
    print(f"Skipped: {skipped}")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare MatSynth for ControlNet training")
    parser.add_argument("--output", type=str, default="./data")
    parser.add_argument("--target", type=str, default="roughness",
                        choices=["roughness", "metallic"])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--all", action="store_true",
                        help="Prepare both roughness and metallic datasets")
    parser.add_argument("--filter-empty", action="store_true",
                        help="Filter out samples with mostly-black target maps (useful for metallic)")

    args = parser.parse_args()

    if args.all:
        for target in ["roughness", "metallic"]:
            # Auto-enable filter for metallic
            filter_empty = args.filter_empty or (target == "metallic")
            prepare_controlnet_dataset(
                args.output, target, args.split, args.max_samples,
                args.resolution, args.num_workers, filter_empty
            )
    else:
        prepare_controlnet_dataset(
            args.output, args.target, args.split, args.max_samples,
            args.resolution, args.num_workers, args.filter_empty
        )


if __name__ == "__main__":
    main()
