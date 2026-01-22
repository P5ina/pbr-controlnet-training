"""
Data preparation script for multi-task PBR training.

Supports:
1. Converting existing PBR dataset to required format
2. Downloading from ambientCG (public domain materials)
3. Random crops and augmentation for seamless textures

Expected output structure:
    data/materials/
        basecolor/
            material_001.jpg
            material_002.jpg
            ...
        normal/
            material_001.jpg
            ...
        roughness/
            material_001.jpg
            ...
        metallic/
            material_001.jpg
            ...
        height/  (optional)
            material_001.jpg
            ...
"""

import argparse
import os
import json
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def convert_existing_dataset(input_dir, output_dir, resolution=512):
    """
    Convert an existing PBR dataset to the required format.

    Expects input structure with materials in subdirectories:
        input_dir/
            MaterialName/
                *_Color.jpg or *_Albedo.jpg or *_BaseColor.jpg
                *_Normal.jpg or *_NormalGL.jpg
                *_Roughness.jpg
                *_Metallic.jpg or *_Metalness.jpg
                *_Height.jpg or *_Displacement.jpg (optional)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directories
    for map_type in ['basecolor', 'normal', 'roughness', 'metallic', 'height']:
        (output_path / map_type).mkdir(parents=True, exist_ok=True)

    # Map file patterns to output types
    patterns = {
        'basecolor': ['Color', 'Albedo', 'BaseColor', 'Diffuse', 'Base_Color'],
        'normal': ['Normal', 'NormalGL', 'Normal_GL', 'NormalDX'],
        'roughness': ['Roughness', 'Rough'],
        'metallic': ['Metallic', 'Metalness', 'Metal'],
        'height': ['Height', 'Displacement', 'Disp', 'Bump'],
    }

    def find_map(material_dir, map_type):
        """Find a map file in the material directory."""
        for pattern in patterns[map_type]:
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                matches = list(material_dir.glob(f'*{pattern}*'))
                matches += list(material_dir.glob(f'*{pattern.lower()}*'))
                matches += list(material_dir.glob(f'*{pattern.upper()}*'))
                if matches:
                    return matches[0]
        return None

    # Find all material directories
    material_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    print(f"Found {len(material_dirs)} material directories")

    converted = 0
    skipped = 0

    for material_dir in tqdm(material_dirs, desc="Converting"):
        material_name = material_dir.name

        # Find all required maps
        maps = {}
        for map_type in ['basecolor', 'normal', 'roughness', 'metallic']:
            map_file = find_map(material_dir, map_type)
            if map_file:
                maps[map_type] = map_file

        # Skip if missing required maps
        if len(maps) < 4:
            skipped += 1
            continue

        # Optional height map
        height_file = find_map(material_dir, 'height')
        if height_file:
            maps['height'] = height_file

        # Process and save maps
        try:
            for map_type, map_file in maps.items():
                img = Image.open(map_file).convert('RGB')
                img = img.resize((resolution, resolution), Image.LANCZOS)

                output_file = output_path / map_type / f"{material_name}.jpg"
                img.save(output_file, 'JPEG', quality=95)

            converted += 1
        except Exception as e:
            print(f"Error processing {material_name}: {e}")
            skipped += 1

    print(f"\nConverted: {converted}, Skipped: {skipped}")


def download_ambientcg(output_dir, resolution=512, max_materials=1000):
    """
    Download PBR materials from ambientCG (CC0 license).

    Note: Requires requests library and internet connection.
    """
    if not HAS_REQUESTS:
        print("Error: requests library not installed. Run: pip install requests")
        return

    output_path = Path(output_dir)

    # Create output directories
    for map_type in ['basecolor', 'normal', 'roughness', 'metallic', 'height']:
        (output_path / map_type).mkdir(parents=True, exist_ok=True)

    # ambientCG API
    api_url = "https://ambientcg.com/api/v2/full_json"
    params = {
        "type": "Material",
        "limit": max_materials,
        "include": "downloadData",
    }

    print("Fetching material list from ambientCG...")
    response = requests.get(api_url, params=params)
    data = response.json()

    materials = data.get("foundAssets", [])
    print(f"Found {len(materials)} materials")

    downloaded = 0

    for material in tqdm(materials, desc="Downloading"):
        asset_id = material.get("assetId", "")

        # Find 1K or 2K download
        downloads = material.get("downloadFolders", {}).get("default", {}).get("downloadFiletypeCategories", {})
        zip_data = None

        for res in ["1K-JPG", "2K-JPG", "1K-PNG", "2K-PNG"]:
            if res in downloads:
                zip_data = downloads[res].get("downloads", [{}])[0]
                break

        if not zip_data:
            continue

        try:
            # Download and extract
            zip_url = zip_data.get("fullDownloadPath", "")
            if not zip_url:
                continue

            # Download ZIP
            import tempfile
            import zipfile

            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                r = requests.get(zip_url, stream=True)
                for chunk in r.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                tmp_path = tmp.name

            # Extract and process
            with zipfile.ZipFile(tmp_path, 'r') as zf:
                names = zf.namelist()

                # Find map files
                map_files = {
                    'basecolor': next((n for n in names if 'Color' in n and n.endswith(('.jpg', '.png'))), None),
                    'normal': next((n for n in names if 'Normal' in n and 'GL' in n and n.endswith(('.jpg', '.png'))), None) or
                             next((n for n in names if 'Normal' in n and n.endswith(('.jpg', '.png'))), None),
                    'roughness': next((n for n in names if 'Roughness' in n and n.endswith(('.jpg', '.png'))), None),
                    'metallic': next((n for n in names if 'Metalness' in n and n.endswith(('.jpg', '.png'))), None),
                    'height': next((n for n in names if 'Displacement' in n and n.endswith(('.jpg', '.png'))), None),
                }

                # Skip if missing required maps (metallic often missing for non-metals)
                if not all(map_files.get(k) for k in ['basecolor', 'normal', 'roughness']):
                    os.unlink(tmp_path)
                    continue

                # Process each map
                for map_type, filename in map_files.items():
                    if not filename:
                        # Create default metallic (black) if missing
                        if map_type == 'metallic':
                            img = Image.new('RGB', (resolution, resolution), (0, 0, 0))
                        else:
                            continue
                    else:
                        with zf.open(filename) as f:
                            img = Image.open(f).convert('RGB')

                    img = img.resize((resolution, resolution), Image.LANCZOS)
                    output_file = output_path / map_type / f"{asset_id}.jpg"
                    img.save(output_file, 'JPEG', quality=95)

            os.unlink(tmp_path)
            downloaded += 1

        except Exception as e:
            print(f"Error downloading {asset_id}: {e}")
            continue

    print(f"\nDownloaded: {downloaded} materials")


def create_random_crops(input_dir, output_dir, crops_per_material=4, crop_size=512):
    """
    Create random crops from larger textures.
    Useful for seamless textures where you want more training data.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directories
    for map_type in ['basecolor', 'normal', 'roughness', 'metallic', 'height']:
        src_dir = input_path / map_type
        dst_dir = output_path / map_type
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            continue

        files = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
        print(f"Processing {len(files)} {map_type} maps...")

        for file in tqdm(files, desc=map_type):
            img = Image.open(file)
            w, h = img.size

            if w < crop_size or h < crop_size:
                # Just copy if too small
                shutil.copy(file, dst_dir / file.name)
                continue

            # Generate random crops
            for i in range(crops_per_material):
                x = np.random.randint(0, w - crop_size + 1)
                y = np.random.randint(0, h - crop_size + 1)

                crop = img.crop((x, y, x + crop_size, y + crop_size))

                output_name = f"{file.stem}_crop{i}{file.suffix}"
                crop.save(dst_dir / output_name, quality=95)


def validate_dataset(data_dir):
    """Validate that the dataset is properly formatted."""
    data_path = Path(data_dir)

    print(f"Validating dataset at {data_path}...\n")

    # Check directories
    required = ['basecolor', 'normal', 'roughness', 'metallic']
    optional = ['height']

    for map_type in required:
        dir_path = data_path / map_type
        if not dir_path.exists():
            print(f"ERROR: Missing required directory: {map_type}/")
            return False

    # Count files
    counts = {}
    for map_type in required + optional:
        dir_path = data_path / map_type
        if dir_path.exists():
            files = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png"))
            counts[map_type] = len(files)
        else:
            counts[map_type] = 0

    print("File counts:")
    for map_type, count in counts.items():
        status = "OK" if count > 0 else "MISSING"
        print(f"  {map_type}: {count} files [{status}]")

    # Check that all required maps have same number of files
    required_counts = [counts[k] for k in required]
    if len(set(required_counts)) != 1:
        print("\nWARNING: Required map directories have different file counts!")
        print("  Make sure each material has all required maps.")

    # Verify filenames match
    basecolor_files = {f.stem for f in (data_path / 'basecolor').glob("*.*") if f.suffix in ['.jpg', '.png']}

    for map_type in required[1:]:
        map_files = {f.stem for f in (data_path / map_type).glob("*.*") if f.suffix in ['.jpg', '.png']}
        missing = basecolor_files - map_files
        extra = map_files - basecolor_files

        if missing:
            print(f"\n{map_type}: Missing {len(missing)} files that exist in basecolor/")
        if extra:
            print(f"\n{map_type}: {len(extra)} extra files not in basecolor/")

    print(f"\nTotal materials: {counts['basecolor']}")
    print("Dataset validation complete!")

    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare data for multi-task PBR training")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Convert existing dataset
    convert_parser = subparsers.add_parser("convert", help="Convert existing PBR dataset")
    convert_parser.add_argument("--input", required=True, help="Input directory")
    convert_parser.add_argument("--output", required=True, help="Output directory")
    convert_parser.add_argument("--resolution", type=int, default=512, help="Output resolution")

    # Download from ambientCG
    download_parser = subparsers.add_parser("download", help="Download from ambientCG")
    download_parser.add_argument("--output", required=True, help="Output directory")
    download_parser.add_argument("--resolution", type=int, default=512, help="Output resolution")
    download_parser.add_argument("--max", type=int, default=1000, help="Max materials to download")

    # Create random crops
    crop_parser = subparsers.add_parser("crop", help="Create random crops from textures")
    crop_parser.add_argument("--input", required=True, help="Input directory")
    crop_parser.add_argument("--output", required=True, help="Output directory")
    crop_parser.add_argument("--crops", type=int, default=4, help="Crops per material")
    crop_parser.add_argument("--size", type=int, default=512, help="Crop size")

    # Validate dataset
    validate_parser = subparsers.add_parser("validate", help="Validate dataset format")
    validate_parser.add_argument("--data", required=True, help="Data directory")

    args = parser.parse_args()

    if args.command == "convert":
        convert_existing_dataset(args.input, args.output, args.resolution)
    elif args.command == "download":
        download_ambientcg(args.output, args.resolution, args.max)
    elif args.command == "crop":
        create_random_crops(args.input, args.output, args.crops, args.size)
    elif args.command == "validate":
        validate_dataset(args.data)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
