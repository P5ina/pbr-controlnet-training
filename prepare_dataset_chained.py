"""
Prepare MatSynth dataset for CHAINED PBR training.
Simple sequential approach - reliable and resumable.
"""

import argparse
from pathlib import Path
import json
import os
from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm


def is_mostly_black(img, threshold=0.05):
    if img is None:
        return True
    arr = np.array(img.convert("L")).astype(float) / 255.0
    return arr.mean() < threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()

    output_path = Path(args.output) / "chained"
    for map_type in ["basecolor", "normal", "roughness", "metallic"]:
        (output_path / map_type).mkdir(parents=True, exist_ok=True)

    print("Loading MatSynth dataset...")
    dataset = load_dataset("gvecchio/MatSynth", split=args.split)

    total = len(dataset)
    if args.max_samples:
        total = min(args.max_samples, total)

    print(f"Processing {total} materials...")

    prompts = {}
    skipped = 0
    metallic_count = 0
    errors = 0

    # Load existing prompts if resuming
    prompts_file = output_path / "prompts_partial.json"
    if prompts_file.exists():
        with open(prompts_file) as f:
            prompts = json.load(f)
        print(f"Resuming: {len(prompts)} already processed")

    pbar = tqdm(range(total), unit="mat")

    for idx in pbar:
        filename = f"{idx:05d}.jpg"
        bc_path = output_path / "basecolor" / filename

        # Skip if already done
        if bc_path.exists():
            continue

        try:
            sample = dataset[idx]

            basecolor = sample.get("basecolor") or sample.get("diffuse")
            normal = sample.get("normal")
            roughness = sample.get("roughness")
            metallic = sample.get("metallic")

            if basecolor is None or normal is None or roughness is None:
                skipped += 1
                continue

            res = args.resolution
            basecolor = basecolor.resize((res, res), Image.BILINEAR).convert("RGB")
            normal = normal.resize((res, res), Image.BILINEAR).convert("RGB")
            roughness = roughness.resize((res, res), Image.BILINEAR).convert("RGB")

            if metallic is None:
                metallic = Image.new("RGB", (res, res), (0, 0, 0))
            else:
                metallic = metallic.resize((res, res), Image.BILINEAR).convert("RGB")
                if not is_mostly_black(metallic):
                    metallic_count += 1

            basecolor.save(bc_path, "JPEG", quality=95)
            normal.save(output_path / "normal" / filename, "JPEG", quality=95)
            roughness.save(output_path / "roughness" / filename, "JPEG", quality=95)
            metallic.save(output_path / "metallic" / filename, "JPEG", quality=95)

            # Caption
            metadata = sample.get("metadata", {}) or {}
            name = str(sample.get("name", f"material_{idx}"))
            category = str(sample.get("category", "unknown"))

            if isinstance(metadata, dict):
                desc = metadata.get("description", "")
                tags = metadata.get("tags", [])
            else:
                desc, tags = "", []

            caption = desc or (", ".join(tags) if tags else f"{name}, {category}")
            prompts[filename] = {"caption": caption, "is_metallic": not is_mostly_black(metallic)}

            # Save progress every 100
            if len(prompts) % 100 == 0:
                with open(prompts_file, "w") as f:
                    json.dump(prompts, f)

        except Exception as e:
            errors += 1
            pbar.set_postfix(errors=errors)
            continue

    # Save final
    with open(output_path / "prompts.json", "w") as f:
        json.dump(prompts, f, indent=2)

    if prompts_file.exists():
        prompts_file.unlink()

    meta = {
        "total_samples": len(prompts),
        "skipped": skipped,
        "errors": errors,
        "metallic_materials": metallic_count,
        "resolution": args.resolution,
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! {len(prompts)} saved, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
