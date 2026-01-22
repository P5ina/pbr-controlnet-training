"""
Prepare MatSynth dataset for CHAINED PBR training.

Creates dataset with ALL maps for each material:
- basecolor (input)
- normal (stage 1 target)
- roughness (stage 2 target)
- metallic (stage 3 target)

This enables chained training where each stage uses previous outputs as conditioning.

Usage:
    python prepare_dataset_chained.py --output ./data --max-samples 4000

Resume from interruption:
    python prepare_dataset_chained.py --output ./data --max-samples 4000 --resume
"""

import os
import argparse
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Thread
import time

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
    img.save(path, "PNG", compress_level=1)


def get_map_from_sample(sample, map_name: str, resolution: int):
    """Extract a specific map from sample with fallback keys."""
    key_variants = [map_name, f"{map_name}_map"]

    for key in key_variants:
        if key in sample and sample[key] is not None:
            return process_image(sample[key], resolution)
    return None


def process_sample(args):
    """Process a single sample and save ALL maps."""
    idx, sample, processed_idx, output_path, resolution = args

    try:
        filename = f"{processed_idx:05d}.png"

        # Check if already processed (for resume)
        if (output_path / "basecolor" / filename).exists():
            # Already done, get metadata
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

            return {
                "filename": filename,
                "caption": caption,
                "is_metallic": False,  # Unknown for skipped
                "skipped": True,
            }

        # Get basecolor (required)
        basecolor = get_map_from_sample(sample, "basecolor", resolution)
        if basecolor is None:
            basecolor = get_map_from_sample(sample, "diffuse", resolution)
        if basecolor is None:
            return None

        # Get normal (required for chaining)
        normal = get_map_from_sample(sample, "normal", resolution)
        if normal is None:
            return None

        # Get roughness (required)
        roughness = get_map_from_sample(sample, "roughness", resolution)
        if roughness is None:
            return None

        # Get metallic (optional - can be mostly black)
        metallic = get_map_from_sample(sample, "metallic", resolution)
        if metallic is None:
            metallic = Image.new("RGB", (resolution, resolution), (0, 0, 0))

        # Save all maps
        save_image_fast(basecolor, output_path / "basecolor" / filename)
        save_image_fast(normal, output_path / "normal" / filename)
        save_image_fast(roughness, output_path / "roughness" / filename)
        save_image_fast(metallic, output_path / "metallic" / filename)

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

        is_metallic = not is_mostly_black(metallic)

        return {
            "filename": filename,
            "caption": caption,
            "is_metallic": is_metallic,
            "skipped": False,
        }

    except Exception as e:
        return None


def prepare_chained_dataset(
    output_dir: str,
    split: str = "train",
    max_samples: int = None,
    resolution: int = 512,
    num_workers: int = 8,
    resume: bool = False,
):
    """Prepare dataset with all PBR maps for chained training."""
    output_path = Path(output_dir) / "chained"

    # Create directories for each map type
    for map_type in ["basecolor", "normal", "roughness", "metallic"]:
        (output_path / map_type).mkdir(parents=True, exist_ok=True)

    print(f"Loading MatSynth dataset ({split} split)...")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Workers: {num_workers}")
    print(f"Output: {output_path}")
    print(f"Resume mode: {resume}")
    print(f"\nMaps to extract: basecolor, normal, roughness, metallic")

    # Load cached dataset
    print("\nLoading dataset from cache...")
    dataset = load_dataset("gvecchio/MatSynth", split=split)

    total_in_dataset = len(dataset)
    total_samples = total_in_dataset if max_samples is None else min(max_samples, total_in_dataset)
    print(f"Dataset loaded: {total_in_dataset} materials")
    print(f"Processing: {total_samples} materials\n")

    prompts = {}
    metallic_count = 0
    skipped = 0
    already_done = 0

    # Use a single ThreadPoolExecutor for all processing
    pbar = tqdm(total=total_samples, unit="mat", smoothing=0.1)

    batch_size = num_workers * 4
    batch = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for idx, sample in enumerate(dataset):
            if idx >= total_samples:
                break

            batch.append((idx, sample, idx, output_path, resolution))

            # Process batch when full
            if len(batch) >= batch_size:
                futures = [executor.submit(process_sample, item) for item in batch]

                for future in as_completed(futures):
                    result = future.result()

                    if result:
                        prompts[result["filename"]] = {
                            "caption": result["caption"],
                            "is_metallic": result["is_metallic"],
                        }
                        if result.get("skipped"):
                            already_done += 1
                        elif result["is_metallic"]:
                            metallic_count += 1
                    else:
                        skipped += 1

                    pbar.update(1)

                batch = []

                # Save progress periodically
                if len(prompts) % 500 == 0 and len(prompts) > 0:
                    with open(output_path / "prompts_partial.json", "w") as f:
                        json.dump(prompts, f)

        # Process remaining batch
        if batch:
            futures = [executor.submit(process_sample, item) for item in batch]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    prompts[result["filename"]] = {
                        "caption": result["caption"],
                        "is_metallic": result["is_metallic"],
                    }
                    if result.get("skipped"):
                        already_done += 1
                    elif result["is_metallic"]:
                        metallic_count += 1
                else:
                    skipped += 1
                pbar.update(1)

    pbar.close()

    # Save final prompts
    with open(output_path / "prompts.json", "w") as f:
        json.dump(prompts, f, indent=2)

    # Remove partial file
    partial_file = output_path / "prompts_partial.json"
    if partial_file.exists():
        partial_file.unlink()

    # Save metadata
    meta = {
        "total_samples": len(prompts),
        "skipped_missing_maps": skipped,
        "already_processed": already_done,
        "metallic_materials": metallic_count,
        "non_metallic_materials": len(prompts) - metallic_count - already_done,
        "resolution": resolution,
        "maps": ["basecolor", "normal", "roughness", "metallic"],
        "chain_order": ["normal", "roughness", "metallic"],
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Chained dataset prepared!")
    print(f"Total materials: {len(prompts)}")
    if already_done > 0:
        print(f"Already processed (resumed): {already_done}")
    print(f"Metallic materials: {metallic_count}")
    print(f"Non-metallic materials: {len(prompts) - metallic_count - already_done}")
    print(f"Skipped (missing maps): {skipped}")
    print(f"\nStructure:")
    print(f"  {output_path}/")
    print(f"    basecolor/  - Input images")
    print(f"    normal/     - Stage 1 target (basecolor → normal)")
    print(f"    roughness/  - Stage 2 target (basecolor + normal → roughness)")
    print(f"    metallic/   - Stage 3 target (basecolor + normal + roughness → metallic)")


def main():
    parser = argparse.ArgumentParser(description="Prepare MatSynth for chained PBR training")
    parser.add_argument("--output", type=str, default="./data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from partial download (skips existing files)")

    args = parser.parse_args()

    prepare_chained_dataset(
        args.output,
        args.split,
        args.max_samples,
        args.resolution,
        args.num_workers,
        args.resume,
    )


if __name__ == "__main__":
    main()
