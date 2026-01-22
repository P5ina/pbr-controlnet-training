"""
Prepare dataset in diffusers ControlNet imagefolder format.

Creates a dataset directory with:
- metadata.jsonl (with image, conditioning_image paths, and text)
- images/ (target PBR maps)
- conditioning_images/ (basecolor inputs)

The diffusers training script is patched to cast conditioning_image
from string paths to actual Image features after loading.
"""

import argparse
from pathlib import Path
import shutil
import json


def prepare_diffusers_dataset(data_dir: str, target: str):
    """Convert chained dataset to diffusers ControlNet imagefolder format."""

    chained_dir = Path(data_dir) / "chained"
    output_dir = Path(data_dir) / "kohya" / target

    # Create directories
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "conditioning_images").mkdir(parents=True, exist_ok=True)

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

    print(f"Converting {len(files)} samples to diffusers imagefolder format...")
    print(f"Target: {target}")
    print(f"Output: {output_dir}")

    metadata = []

    for i, bc_file in enumerate(files):
        filename = bc_file.name
        target_file = target_dir / filename

        if not target_file.exists():
            continue

        # New filename
        new_name = f"{i:05d}.jpg"

        # Copy target image (what model should generate) -> images/
        shutil.copy(target_file, output_dir / "images" / new_name)

        # Copy conditioning image (basecolor input) -> conditioning_images/
        shutil.copy(bc_file, output_dir / "conditioning_images" / new_name)

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

        # For imagefolder format:
        # - file_name -> loaded as 'image' column (auto-converted to Image)
        # - conditioning_image -> absolute path (cast_column needs full paths)
        # - text -> caption
        cond_image_path = (output_dir / "conditioning_images" / new_name).absolute()
        metadata.append({
            "file_name": f"images/{new_name}",
            "conditioning_image": str(cond_image_path),
            "text": full_prompt
        })

        if (i + 1) % 500 == 0:
            print(f"  Copied {i + 1}/{len(files)} files...")

    # Write metadata.jsonl
    with open(output_dir / "metadata.jsonl", "w") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")

    print(f"\nDone! Created {len(metadata)} samples")
    print(f"\nStructure:")
    print(f"  {output_dir}/")
    print(f"    metadata.jsonl          - Links images with captions")
    print(f"    images/                 - Target images ({target} maps)")
    print(f"    conditioning_images/    - Basecolor inputs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--target", type=str, required=True,
                        choices=["normal", "roughness", "metallic"])
    args = parser.parse_args()

    prepare_diffusers_dataset(args.data_dir, args.target)


if __name__ == "__main__":
    main()
