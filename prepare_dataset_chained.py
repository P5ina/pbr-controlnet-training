"""
Prepare MatSynth dataset for CHAINED PBR training.

Creates dataset with ALL maps for each material:
- basecolor (input)
- normal (stage 1 target)
- roughness (stage 2 target)
- metallic (stage 3 target)

Usage:
    python prepare_dataset_chained.py --output ./data --max-samples 4000
"""

import argparse
from pathlib import Path
import json

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def is_mostly_black(img: Image.Image, threshold: float = 0.05) -> bool:
    if img is None:
        return True
    arr = np.array(img.convert("L")).astype(float) / 255.0
    return arr.mean() < threshold


def prepare_chained_dataset(
    output_dir: str,
    split: str = "train",
    max_samples: int = None,
    resolution: int = 512,
    num_workers: int = 8,
):
    output_path = Path(output_dir) / "chained"

    for map_type in ["basecolor", "normal", "roughness", "metallic"]:
        (output_path / map_type).mkdir(parents=True, exist_ok=True)

    print(f"Loading MatSynth dataset ({split} split)...")
    dataset = load_dataset("gvecchio/MatSynth", split=split)

    total_in_dataset = len(dataset)
    total_samples = total_in_dataset if max_samples is None else min(max_samples, total_in_dataset)
    print(f"Dataset: {total_in_dataset} materials, processing: {total_samples}")
    print(f"Output: {output_path}")
    print(f"Workers: {num_workers}\n")

    prompts = {}
    metallic_count = 0
    skipped = 0

    pbar = tqdm(total=total_samples, unit="mat")

    for idx in range(total_samples):
        sample = dataset[idx]
        filename = f"{idx:05d}.png"

        # Skip if already exists
        if (output_path / "basecolor" / filename).exists():
            pbar.update(1)
            continue

        try:
            # Get maps
            basecolor = sample.get("basecolor") or sample.get("diffuse")
            normal = sample.get("normal")
            roughness = sample.get("roughness")
            metallic = sample.get("metallic")

            if basecolor is None or normal is None or roughness is None:
                skipped += 1
                pbar.update(1)
                continue

            # Process and save
            basecolor = basecolor.resize((resolution, resolution), Image.BILINEAR).convert("RGB")
            normal = normal.resize((resolution, resolution), Image.BILINEAR).convert("RGB")
            roughness = roughness.resize((resolution, resolution), Image.BILINEAR).convert("RGB")

            if metallic is None:
                metallic = Image.new("RGB", (resolution, resolution), (0, 0, 0))
            else:
                metallic = metallic.resize((resolution, resolution), Image.BILINEAR).convert("RGB")
                if not is_mostly_black(metallic):
                    metallic_count += 1

            basecolor.save(output_path / "basecolor" / filename, "PNG", compress_level=0)
            normal.save(output_path / "normal" / filename, "PNG", compress_level=0)
            roughness.save(output_path / "roughness" / filename, "PNG", compress_level=0)
            metallic.save(output_path / "metallic" / filename, "PNG", compress_level=0)

            # Caption
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

            prompts[filename] = {"caption": caption, "is_metallic": not is_mostly_black(metallic)}

        except Exception as e:
            skipped += 1

        pbar.update(1)

        # Save progress
        if len(prompts) % 500 == 0 and len(prompts) > 0:
            with open(output_path / "prompts_partial.json", "w") as f:
                json.dump(prompts, f)

    pbar.close()

    # Save final
    with open(output_path / "prompts.json", "w") as f:
        json.dump(prompts, f, indent=2)

    partial = output_path / "prompts_partial.json"
    if partial.exists():
        partial.unlink()

    meta = {
        "total_samples": len(prompts),
        "skipped": skipped,
        "metallic_materials": metallic_count,
        "resolution": resolution,
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Done! {len(prompts)} materials, {skipped} skipped")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    prepare_chained_dataset(
        args.output, args.split, args.max_samples, args.resolution, args.num_workers
    )


if __name__ == "__main__":
    main()
