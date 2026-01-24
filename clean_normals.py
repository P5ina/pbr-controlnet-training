#!/usr/bin/env python3
"""
Remove bad normal maps and their corresponding PBR files.
"""

import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import shutil


def is_bad_normal(image_path):
    """Check if a normal map is bad format."""
    try:
        img = Image.open(image_path).convert('RGB')
        arr = np.array(img).astype(np.float32)

        r_mean = arr[:, :, 0].mean()
        g_mean = arr[:, :, 1].mean()
        b_mean = arr[:, :, 2].mean()

        # Bad if blue is not dominant
        if b_mean < r_mean or b_mean < g_mean:
            return True, f"Blue not dominant (R={r_mean:.0f}, G={g_mean:.0f}, B={b_mean:.0f})"

        # Bad if blue is too low
        if b_mean < 100:
            return True, f"Blue too low ({b_mean:.0f})"

        return False, "OK"
    except Exception as e:
        return True, f"Error: {e}"


def main():
    parser = argparse.ArgumentParser(description='Remove bad normal maps and corresponding files')
    parser.add_argument('--data-dir', type=str, default='./data/materials',
                        help='Materials directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only show what would be deleted, don\'t actually delete')
    parser.add_argument('--quarantine', type=str, default='',
                        help='Move bad files to this directory instead of deleting')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    normal_dir = data_dir / 'normal'

    if not normal_dir.exists():
        print(f"Error: {normal_dir} does not exist")
        return

    # Find all normal maps
    extensions = ['*.jpg', '*.jpeg', '*.png']
    files = []
    for ext in extensions:
        files.extend(normal_dir.glob(ext))
    files = sorted(files)

    print(f"Checking {len(files)} normal maps...")

    bad_files = []
    for f in files:
        is_bad, reason = is_bad_normal(f)
        if is_bad:
            bad_files.append((f, reason))

    print(f"\nFound {len(bad_files)} bad normal maps ({100*len(bad_files)/len(files):.1f}%)")

    if not bad_files:
        print("No bad files to remove!")
        return

    # Show bad files
    print("\nBad files:")
    for f, reason in bad_files[:20]:
        print(f"  {f.name}: {reason}")
    if len(bad_files) > 20:
        print(f"  ... and {len(bad_files) - 20} more")

    # Setup quarantine dir if specified
    quarantine_dir = None
    if args.quarantine:
        quarantine_dir = Path(args.quarantine)
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ['basecolor', 'normal', 'roughness', 'metallic', 'height']:
            (quarantine_dir / subdir).mkdir(exist_ok=True)

    # Remove bad files and corresponding PBR maps
    subdirs = ['basecolor', 'normal', 'roughness', 'metallic', 'height']
    removed_count = 0

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Removing bad materials...")

    for bad_normal, reason in bad_files:
        base_name = bad_normal.stem

        for subdir in subdirs:
            subdir_path = data_dir / subdir
            if not subdir_path.exists():
                continue

            # Find matching file (could be jpg or png)
            for ext in ['.jpg', '.jpeg', '.png']:
                file_path = subdir_path / (base_name + ext)
                if file_path.exists():
                    if args.dry_run:
                        print(f"  Would remove: {file_path}")
                    elif quarantine_dir:
                        dest = quarantine_dir / subdir / file_path.name
                        shutil.move(str(file_path), str(dest))
                        print(f"  Moved: {file_path.name} -> quarantine")
                    else:
                        file_path.unlink()
                        print(f"  Removed: {file_path}")
                    removed_count += 1
                    break

    print(f"\n{'Would remove' if args.dry_run else 'Removed'} {removed_count} files total")
    print(f"Remaining materials: ~{len(files) - len(bad_files)}")


if __name__ == '__main__':
    main()
