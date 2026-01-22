"""
Download and prepare MatSynth dataset for multi-task PBR training.

MatSynth contains ~4000 high-quality PBR materials with:
- Basecolor, Normal, Roughness, Metallic, Height maps
- CC0 and permissive licenses

https://huggingface.co/datasets/gvecchio/MatSynth
"""

import argparse
from pathlib import Path
from PIL import Image
import io
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def download_matsynth(output_dir, resolution=512, max_materials=None, cc0_only=False, num_workers=4):
    """Download MatSynth and convert to training format."""

    if not HAS_DATASETS:
        print("Error: 'datasets' library not installed.")
        print("Run: pip install datasets")
        return

    output_path = Path(output_dir)

    # Create output directories
    for map_type in ['basecolor', 'normal', 'roughness', 'metallic', 'height']:
        (output_path / map_type).mkdir(parents=True, exist_ok=True)

    print("Loading MatSynth dataset from Hugging Face...")
    print("(This may take a while on first run as it downloads the dataset)")

    # Load dataset
    ds = load_dataset("gvecchio/MatSynth", split="train", streaming=True)

    # Filter CC0 only if requested
    if cc0_only:
        print("Filtering for CC0 licensed materials only...")
        ds = ds.filter(lambda x: x["metadata"]["license"] == "CC0")

    # Keep only the maps we need
    ds = ds.select_columns(["metadata", "basecolor", "normal", "roughness", "metallic", "height"])

    processed = 0
    skipped = 0

    print(f"\nDownloading materials to {output_path}...")
    print(f"Resolution: {resolution}x{resolution}")
    if max_materials:
        print(f"Max materials: {max_materials}")

    for item in tqdm(ds, desc="Processing"):
        if max_materials and processed >= max_materials:
            break

        try:
            # Get material name from metadata
            metadata = item.get("metadata", {})
            name = metadata.get("name", f"material_{processed:05d}")
            # Clean name for filesystem
            name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

            # Process each map
            maps_saved = 0
            for map_type in ['basecolor', 'normal', 'roughness', 'metallic', 'height']:
                img_data = item.get(map_type)

                if img_data is None:
                    continue

                # Handle PIL Image or bytes
                if isinstance(img_data, Image.Image):
                    img = img_data
                elif isinstance(img_data, bytes):
                    img = Image.open(io.BytesIO(img_data))
                elif isinstance(img_data, dict) and 'bytes' in img_data:
                    img = Image.open(io.BytesIO(img_data['bytes']))
                else:
                    continue

                # Convert to RGB (some maps might be grayscale)
                if map_type in ['roughness', 'metallic', 'height']:
                    # These are grayscale, but save as RGB for consistency
                    if img.mode != 'RGB':
                        if img.mode == 'L':
                            img = img.convert('RGB')
                        elif img.mode == 'I;16':
                            # 16-bit to 8-bit
                            import numpy as np
                            arr = np.array(img).astype(np.float32)
                            arr = (arr / arr.max() * 255).astype(np.uint8)
                            img = Image.fromarray(arr).convert('RGB')
                        else:
                            img = img.convert('RGB')
                else:
                    img = img.convert('RGB')

                # Resize
                img = img.resize((resolution, resolution), Image.LANCZOS)

                # Save
                output_file = output_path / map_type / f"{name}.jpg"
                img.save(output_file, 'JPEG', quality=95)
                maps_saved += 1

            # Only count if we saved all required maps
            if maps_saved >= 4:  # basecolor, normal, roughness, metallic minimum
                processed += 1
            else:
                skipped += 1
                # Clean up partial saves
                for map_type in ['basecolor', 'normal', 'roughness', 'metallic', 'height']:
                    f = output_path / map_type / f"{name}.jpg"
                    if f.exists():
                        f.unlink()

        except Exception as e:
            print(f"\nError processing material: {e}")
            skipped += 1
            continue

    print(f"\n{'='*50}")
    print(f"Download complete!")
    print(f"  Processed: {processed} materials")
    print(f"  Skipped: {skipped}")
    print(f"  Output: {output_path}")
    print(f"{'='*50}")

    # Validate
    print("\nValidating dataset...")
    for map_type in ['basecolor', 'normal', 'roughness', 'metallic', 'height']:
        count = len(list((output_path / map_type).glob("*.jpg")))
        print(f"  {map_type}: {count} files")


def main():
    parser = argparse.ArgumentParser(description="Download MatSynth dataset")
    parser.add_argument("--output", type=str, default="./data/materials", help="Output directory")
    parser.add_argument("--resolution", type=int, default=512, help="Output resolution")
    parser.add_argument("--max", type=int, default=None, help="Max materials to download (default: all)")
    parser.add_argument("--cc0-only", action="store_true", help="Only download CC0 licensed materials")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    download_matsynth(
        args.output,
        resolution=args.resolution,
        max_materials=args.max,
        cc0_only=args.cc0_only,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
