"""
Prepare dataset in diffusers ControlNet format.

Creates a HuggingFace Dataset with proper Image features for both
target images and conditioning images.
"""

import argparse
from pathlib import Path
import json

from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage


def prepare_diffusers_dataset(data_dir: str, target: str):
    """Convert chained dataset to diffusers ControlNet format."""

    chained_dir = Path(data_dir) / "chained"
    output_dir = Path(data_dir) / "kohya" / target / "train"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts_file = chained_dir / "prompts.json"
    if prompts_file.exists():
        with open(prompts_file) as f:
            prompts = json.load(f)
    else:
        prompts = {}

    # Get all files
    basecolor_dir = chained_dir / "basecolor"
    target_dir = chained_dir / target

    files = sorted([f for f in basecolor_dir.iterdir() if f.suffix in ['.jpg', '.png']])

    print(f"Converting {len(files)} samples to diffusers format...")
    print(f"Target: {target}")
    print(f"Output: {output_dir}")

    # Build dataset records
    records = []
    for i, bc_file in enumerate(files):
        filename = bc_file.name
        target_file = target_dir / filename

        if not target_file.exists():
            continue

        # Get caption
        prompt_data = prompts.get(filename, {})
        caption = prompt_data.get("caption", "material texture")

        # Create specific prompt for target type
        if target == "normal":
            full_prompt = f"normal map, {caption}, blue purple normal map, pbr texture, seamless tileable"
        elif target == "roughness":
            full_prompt = f"roughness map, {caption}, grayscale roughness texture, pbr material, seamless tileable"
        elif target == "metallic":
            full_prompt = f"metallic map, {caption}, grayscale metallic texture, pbr material, seamless tileable"
        else:
            full_prompt = f"{target} map, {caption}, pbr texture"

        records.append({
            "image": str(target_file.absolute()),
            "conditioning_image": str(bc_file.absolute()),
            "text": full_prompt
        })

        if (i + 1) % 500 == 0:
            print(f"  Prepared {i + 1}/{len(files)} records...")

    print(f"\nCreating HuggingFace Dataset with {len(records)} samples...")

    # Create dataset with proper features
    features = Features({
        "image": Image(),
        "conditioning_image": Image(),
        "text": Value("string")
    })

    dataset = Dataset.from_dict({
        "image": [r["image"] for r in records],
        "conditioning_image": [r["conditioning_image"] for r in records],
        "text": [r["text"] for r in records],
    }, features=features)

    # Save to disk
    dataset.save_to_disk(str(output_dir))

    print(f"\nDone! Created dataset with {len(dataset)} samples")
    print(f"Saved to: {output_dir}")
    print(f"\nFeatures:")
    print(f"  image: Image (target PBR map)")
    print(f"  conditioning_image: Image (basecolor input)")
    print(f"  text: string (prompt)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--target", type=str, required=True,
                        choices=["normal", "roughness", "metallic"])
    args = parser.parse_args()

    prepare_diffusers_dataset(args.data_dir, args.target)


if __name__ == "__main__":
    main()
