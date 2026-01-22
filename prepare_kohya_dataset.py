"""
Prepare dataset in diffusers ControlNet format.

Creates a dataset directory with:
- metadata.jsonl
- images/ (target PBR maps)
- conditioning_images/ (basecolor inputs)
- pbr_controlnet.py (custom loader script)
"""

import argparse
from pathlib import Path
import shutil
import json


LOADER_SCRIPT = '''"""Custom dataset loader for PBR ControlNet training."""
import os
import json
import datasets

_FEATURES = datasets.Features({
    "image": datasets.Image(),
    "conditioning_image": datasets.Image(),
    "text": datasets.Value("string"),
})


class PBRControlNetDataset(datasets.GeneratorBasedBuilder):
    """PBR ControlNet dataset with conditioning images."""

    BUILDER_CONFIGS = [datasets.BuilderConfig(name="default", version=datasets.Version("1.0.0"))]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            features=_FEATURES,
            supervised_keys=("conditioning_image", "text"),
        )

    def _split_generators(self, dl_manager):
        base_path = os.path.dirname(os.path.abspath(__file__))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metadata_path": os.path.join(base_path, "metadata.jsonl"),
                    "images_dir": os.path.join(base_path, "images"),
                    "conditioning_dir": os.path.join(base_path, "conditioning_images"),
                },
            ),
        ]

    def _generate_examples(self, metadata_path, images_dir, conditioning_dir):
        with open(metadata_path, "r") as f:
            for idx, line in enumerate(f):
                row = json.loads(line)
                image_path = os.path.join(images_dir, row["image"])
                cond_path = os.path.join(conditioning_dir, row["conditioning_image"])

                yield idx, {
                    "image": image_path,
                    "conditioning_image": cond_path,
                    "text": row["text"],
                }
'''


def prepare_diffusers_dataset(data_dir: str, target: str):
    """Convert chained dataset to diffusers ControlNet format."""

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

    print(f"Converting {len(files)} samples to diffusers format...")
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

        metadata.append({
            "image": new_name,
            "conditioning_image": new_name,
            "text": full_prompt
        })

        if (i + 1) % 500 == 0:
            print(f"  Copied {i + 1}/{len(files)} files...")

    # Write metadata.jsonl
    with open(output_dir / "metadata.jsonl", "w") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")

    # Write custom loader script (must match directory name for HuggingFace to find it)
    with open(output_dir / f"{target}.py", "w") as f:
        f.write(LOADER_SCRIPT)

    print(f"\nDone! Created {len(metadata)} samples")
    print(f"\nStructure:")
    print(f"  {output_dir}/")
    print(f"    metadata.jsonl")
    print(f"    {target}.py        - Custom loader script")
    print(f"    images/            - Target images ({target} maps)")
    print(f"    conditioning_images/ - Basecolor inputs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--target", type=str, required=True,
                        choices=["normal", "roughness", "metallic"])
    args = parser.parse_args()

    prepare_diffusers_dataset(args.data_dir, args.target)


if __name__ == "__main__":
    main()
