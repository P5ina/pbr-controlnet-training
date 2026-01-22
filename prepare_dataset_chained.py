"""
Prepare MatSynth dataset for CHAINED PBR training.
Uses HuggingFace datasets native multiprocessing for speed.
"""

import argparse
from pathlib import Path
import json
from datasets import load_dataset
from PIL import Image
import numpy as np


def is_mostly_black(img, threshold=0.05):
    if img is None:
        return True
    arr = np.array(img.convert("L")).astype(float) / 255.0
    return arr.mean() < threshold


def process_and_save(example, idx, output_dir, resolution):
    """Process single example - called by dataset.map()"""
    import os
    filename = f"{idx:05d}.jpg"

    bc_path = os.path.join(output_dir, "basecolor", filename)
    if os.path.exists(bc_path):
        return {"success": True, "filename": filename, "skipped": True}

    try:
        basecolor = example.get("basecolor") or example.get("diffuse")
        normal = example.get("normal")
        roughness = example.get("roughness")
        metallic = example.get("metallic")

        if basecolor is None or normal is None or roughness is None:
            return {"success": False}

        basecolor = basecolor.resize((resolution, resolution), Image.BILINEAR).convert("RGB")
        normal = normal.resize((resolution, resolution), Image.BILINEAR).convert("RGB")
        roughness = roughness.resize((resolution, resolution), Image.BILINEAR).convert("RGB")

        if metallic is None:
            metallic = Image.new("RGB", (resolution, resolution), (0, 0, 0))
        else:
            metallic = metallic.resize((resolution, resolution), Image.BILINEAR).convert("RGB")

        basecolor.save(bc_path, "JPEG", quality=95)
        normal.save(os.path.join(output_dir, "normal", filename), "JPEG", quality=95)
        roughness.save(os.path.join(output_dir, "roughness", filename), "JPEG", quality=95)
        metallic.save(os.path.join(output_dir, "metallic", filename), "JPEG", quality=95)

        metadata = example.get("metadata", {}) or {}
        name = str(example.get("name", f"material_{idx}"))
        category = str(example.get("category", "unknown"))

        if isinstance(metadata, dict):
            desc = metadata.get("description", "")
            tags = metadata.get("tags", [])
        else:
            desc, tags = "", []

        caption = desc or (", ".join(tags) if tags else f"{name}, {category}")

        return {
            "success": True,
            "filename": filename,
            "caption": caption,
            "is_metallic": not is_mostly_black(metallic),
            "skipped": False
        }
    except:
        return {"success": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=32)
    args = parser.parse_args()

    output_path = Path(args.output) / "chained"
    for map_type in ["basecolor", "normal", "roughness", "metallic"]:
        (output_path / map_type).mkdir(parents=True, exist_ok=True)

    print(f"Loading MatSynth dataset...")
    dataset = load_dataset("gvecchio/MatSynth", split=args.split)

    total = len(dataset)
    if args.max_samples:
        total = min(args.max_samples, total)
        dataset = dataset.select(range(total))

    print(f"Processing {total} materials with {args.num_workers} workers...")

    # Use dataset.map with multiprocessing
    output_dir = str(output_path)  # Convert to string for multiprocessing
    results = dataset.map(
        lambda example, idx: process_and_save(example, idx, output_dir, args.resolution),
        with_indices=True,
        num_proc=args.num_workers,
        desc="Processing"
    )

    # Collect results
    prompts = {}
    skipped = 0
    metallic_count = 0

    for r in results:
        if r.get("success") and r.get("filename"):
            if not r.get("skipped"):
                prompts[r["filename"]] = {
                    "caption": r.get("caption", ""),
                    "is_metallic": r.get("is_metallic", False)
                }
                if r.get("is_metallic"):
                    metallic_count += 1
        else:
            skipped += 1

    with open(output_path / "prompts.json", "w") as f:
        json.dump(prompts, f, indent=2)

    meta = {
        "total_samples": len(prompts),
        "skipped": skipped,
        "metallic_materials": metallic_count,
        "resolution": args.resolution,
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! {len(prompts)} materials saved, {skipped} skipped")


if __name__ == "__main__":
    main()
