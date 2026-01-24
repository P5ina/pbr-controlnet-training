#!/usr/bin/env python3
"""
Check normal maps for incorrect format.

Valid normal maps should:
- Be predominantly blue (high B channel for Z-up convention)
- Have R,G centered around 128 (0.5) for flat areas
- B channel should be > R and > G on average

This script flags:
- Images where blue is not dominant
- Images with wrong color space (green/cyan dominant)
- Images that might be DirectX format (inverted Y)
"""

import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict


def analyze_normal_map(image_path):
    """Analyze a normal map and return statistics."""
    try:
        img = Image.open(image_path).convert('RGB')
        arr = np.array(img).astype(np.float32)

        r_mean = arr[:, :, 0].mean()
        g_mean = arr[:, :, 1].mean()
        b_mean = arr[:, :, 2].mean()

        r_std = arr[:, :, 0].std()
        g_std = arr[:, :, 1].std()
        b_std = arr[:, :, 2].std()

        # Check which channel is dominant
        max_channel = np.argmax([r_mean, g_mean, b_mean])
        channel_names = ['RED', 'GREEN', 'BLUE']
        dominant = channel_names[max_channel]

        # For valid normal maps, blue should be highest (Z-up)
        # and R,G should be around 128 for mostly flat surfaces

        issues = []

        # Blue should be dominant
        if b_mean < r_mean or b_mean < g_mean:
            issues.append(f"Blue not dominant (R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f})")

        # Blue should be > 128 (positive Z)
        if b_mean < 100:
            issues.append(f"Blue too low ({b_mean:.1f}) - might be inverted or wrong format")

        # Green-dominant often indicates wrong format
        if g_mean > b_mean and g_mean > 150:
            issues.append(f"Green dominant ({g_mean:.1f}) - possibly wrong color space")

        # Cyan (high G + high B, low R) is suspicious
        if g_mean > 150 and b_mean > 150 and r_mean < 100:
            issues.append(f"Cyan-ish - unusual for normal maps")

        # Red dominant is very wrong
        if r_mean > b_mean and r_mean > g_mean and r_mean > 150:
            issues.append(f"Red dominant ({r_mean:.1f}) - definitely wrong format")

        return {
            'path': image_path,
            'r_mean': r_mean,
            'g_mean': g_mean,
            'b_mean': b_mean,
            'r_std': r_std,
            'g_std': g_std,
            'b_std': b_std,
            'dominant': dominant,
            'issues': issues,
            'valid': len(issues) == 0
        }
    except Exception as e:
        return {
            'path': image_path,
            'issues': [f"Error loading: {e}"],
            'valid': False
        }


def main():
    parser = argparse.ArgumentParser(description='Check normal maps for format issues')
    parser.add_argument('--dir', type=str, default='./data/materials/normal',
                        help='Directory containing normal maps')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show all files, not just problematic ones')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of files to check (0 = all)')
    args = parser.parse_args()

    normal_dir = Path(args.dir)
    if not normal_dir.exists():
        print(f"Error: Directory {normal_dir} does not exist")
        return

    # Find all image files
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    files = []
    for ext in extensions:
        files.extend(normal_dir.glob(ext))
    files = sorted(files)

    if args.limit > 0:
        files = files[:args.limit]

    print(f"Checking {len(files)} normal maps in {normal_dir}...")
    print()

    results = []
    issues_by_type = defaultdict(list)

    for i, f in enumerate(files):
        result = analyze_normal_map(f)
        results.append(result)

        if args.verbose or not result['valid']:
            status = "OK" if result['valid'] else "ISSUES"
            print(f"[{status}] {f.name}")
            if 'r_mean' in result:
                print(f"       R={result['r_mean']:.1f} G={result['g_mean']:.1f} B={result['b_mean']:.1f} (dominant: {result['dominant']})")
            for issue in result.get('issues', []):
                print(f"       ! {issue}")
                issues_by_type[issue.split('(')[0].strip()].append(f.name)
            if not result['valid']:
                print()

        # Progress
        if (i + 1) % 100 == 0:
            print(f"... checked {i + 1}/{len(files)}")

    # Summary
    valid_count = sum(1 for r in results if r['valid'])
    invalid_count = len(results) - valid_count

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files:    {len(results)}")
    print(f"Valid:          {valid_count} ({100*valid_count/len(results):.1f}%)")
    print(f"With issues:    {invalid_count} ({100*invalid_count/len(results):.1f}%)")
    print()

    if issues_by_type:
        print("Issues breakdown:")
        for issue_type, files_list in sorted(issues_by_type.items(), key=lambda x: -len(x[1])):
            print(f"  {issue_type}: {len(files_list)} files")
        print()

        # Show some examples of problematic files
        print("Example problematic files:")
        shown = 0
        for r in results:
            if not r['valid'] and shown < 10:
                print(f"  - {r['path'].name}")
                shown += 1
    else:
        print("All normal maps appear to be in correct format!")

    # Calculate average stats for valid maps
    valid_results = [r for r in results if r['valid'] and 'r_mean' in r]
    if valid_results:
        avg_r = np.mean([r['r_mean'] for r in valid_results])
        avg_g = np.mean([r['g_mean'] for r in valid_results])
        avg_b = np.mean([r['b_mean'] for r in valid_results])
        print()
        print(f"Average RGB for valid maps: R={avg_r:.1f} G={avg_g:.1f} B={avg_b:.1f}")


if __name__ == '__main__':
    main()
