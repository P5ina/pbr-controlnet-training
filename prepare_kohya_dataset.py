"""
Prepare dataset in kohya-ss format for ControlNet training.

Kohya format:
    data/kohya/{target}/
        image/
            0001.jpg  (target image - what we want to generate)
            0002.jpg
            ...
        conditioning/
            0001.jpg  (conditioning image - basecolor)
            0002.jpg
            ...
        0001.txt  (caption for 0001.jpg)
        0002.txt
        ...
"""

import argparse
from pathlib import Path
import shutil
import json


def prepare_kohya_dataset(data_dir: str, target: str):
    """Convert chained dataset to kohya format."""

    chained_dir = Path(data_dir) / "chained"
    kohya_dir = Path(data_dir) / "kohya" / target

    # Create directories
    (kohya_dir / "image").mkdir(parents=True, exist_ok=True)
    (kohya_dir / "conditioning").mkdir(parents=True, exist_ok=True)

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

    print(f"Converting {len(files)} samples to kohya format...")
    print(f"Target: {target}")
    print(f"Output: {kohya_dir}")

    for i, bc_file in enumerate(files):
        filename = bc_file.name
        stem = bc_file.stem

        target_file = target_dir / filename

        if not target_file.exists():
            continue

        # New filename with index for kohya (simpler naming)
        new_name = f"{i:05d}.jpg"

        # Copy target image (what model should generate)
        shutil.copy(target_file, kohya_dir / "image" / new_name)

        # Copy conditioning image (basecolor input)
        shutil.copy(bc_file, kohya_dir / "conditioning" / new_name)

        # Create caption file
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

        caption_file = kohya_dir / f"{i:05d}.txt"
        with open(caption_file, "w") as f:
            f.write(full_prompt)

    print(f"\nDone! Created {len(files)} samples")
    print(f"\nStructure:")
    print(f"  {kohya_dir}/")
    print(f"    image/           - Target images (what model generates)")
    print(f"    conditioning/    - Input images (basecolor)")
    print(f"    *.txt           - Captions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--target", type=str, required=True,
                        choices=["normal", "roughness", "metallic"])
    args = parser.parse_args()

    prepare_kohya_dataset(args.data_dir, args.target)


if __name__ == "__main__":
    main()
