"""
Analyze color shifts in generated images across different alpha values.

This script loads images generated with different alpha values and computes
various color metrics to quantify how the weight vector affects image colors.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from skimage import color
import seaborn as sns


def load_images_by_seed_alpha(image_dir: str, pattern: str = "seed*_alpha*_styles*.png") -> Dict:
    """
    Load all generated images and organize by seed and alpha.

    Returns:
        Dict with structure: {seed: {alpha: image_array}}
    """
    image_dir = Path(image_dir)
    images = {}

    # Pattern to extract seed, alpha, and style range from filename
    regex = re.compile(r'seed(\d+)_alpha(-?\d+)_styles(\d+)-(\d+)\.png')

    for img_path in sorted(image_dir.glob(pattern)):
        match = regex.match(img_path.name)
        if match:
            seed = int(match.group(1))
            alpha = int(match.group(2))
            style_start = int(match.group(3))
            style_end = int(match.group(4))

            # Load image
            img = np.array(Image.open(img_path))

            if seed not in images:
                images[seed] = {}
            images[seed][alpha] = {
                'image': img,
                'style_range': (style_start, style_end),
                'path': str(img_path)
            }

    return images


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to LAB color space."""
    # Normalize to [0, 1] range
    img_normalized = image.astype(np.float32) / 255.0
    # Convert to LAB
    lab = color.rgb2lab(img_normalized)
    return lab


def compute_color_statistics(image: np.ndarray) -> Dict:
    """
    Compute comprehensive color statistics for an image.

    Returns dict with:
        - RGB mean, std, skewness, kurtosis per channel
        - LAB mean, std per channel
        - Overall luminance statistics
    """
    stats_dict = {}

    # RGB statistics
    for i, channel in enumerate(['R', 'G', 'B']):
        channel_data = image[:, :, i].flatten()
        stats_dict[f'rgb_{channel}_mean'] = np.mean(channel_data)
        stats_dict[f'rgb_{channel}_std'] = np.std(channel_data)
        stats_dict[f'rgb_{channel}_skewness'] = stats.skew(channel_data)
        stats_dict[f'rgb_{channel}_kurtosis'] = stats.kurtosis(channel_data)

    # LAB statistics
    lab = rgb_to_lab(image)
    for i, channel in enumerate(['L', 'A', 'B']):
        channel_data = lab[:, :, i].flatten()
        stats_dict[f'lab_{channel}_mean'] = np.mean(channel_data)
        stats_dict[f'lab_{channel}_std'] = np.std(channel_data)

    # Overall statistics
    stats_dict['overall_brightness'] = np.mean(image)
    stats_dict['overall_saturation'] = np.std(image)

    return stats_dict


def compute_delta_e(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute average Delta E (CIE76) between two images.
    This measures perceptual color difference.
    """
    lab1 = rgb_to_lab(img1)
    lab2 = rgb_to_lab(img2)

    # Delta E formula: sqrt((L2-L1)^2 + (a2-a1)^2 + (b2-b1)^2)
    delta_e = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=2))

    return np.mean(delta_e)


def compute_color_histogram(image: np.ndarray, bins: int = 50) -> Dict:
    """Compute color histograms for each channel."""
    histograms = {}

    for i, channel in enumerate(['R', 'G', 'B']):
        hist, bin_edges = np.histogram(image[:, :, i].flatten(), bins=bins, range=(0, 255))
        histograms[channel] = {'hist': hist, 'bins': bin_edges}

    return histograms


def analyze_seed_images(seed_data: Dict, reference_alpha: int = 0) -> Dict:
    """
    Analyze all images for a single seed across different alphas.

    Args:
        seed_data: Dict mapping alpha -> image data
        reference_alpha: Alpha value to use as reference (default: 0)

    Returns:
        Dict with analysis results
    """
    results = {
        'alphas': [],
        'delta_e': [],
        'rgb_mean_shift': [],
        'lab_mean_shift': [],
        'brightness_shift': [],
        'color_stats': []
    }

    # Get reference image
    if reference_alpha not in seed_data:
        # Use the alpha closest to 0
        reference_alpha = min(seed_data.keys(), key=lambda x: abs(x))

    ref_img = seed_data[reference_alpha]['image']
    ref_stats = compute_color_statistics(ref_img)

    # Analyze each alpha
    for alpha in sorted(seed_data.keys()):
        img = seed_data[alpha]['image']
        stats_current = compute_color_statistics(img)

        results['alphas'].append(alpha)
        results['color_stats'].append(stats_current)

        # Compute Delta E
        delta_e = compute_delta_e(ref_img, img)
        results['delta_e'].append(delta_e)

        # RGB mean shift
        rgb_shift = np.sqrt(
            (stats_current['rgb_R_mean'] - ref_stats['rgb_R_mean'])**2 +
            (stats_current['rgb_G_mean'] - ref_stats['rgb_G_mean'])**2 +
            (stats_current['rgb_B_mean'] - ref_stats['rgb_B_mean'])**2
        )
        results['rgb_mean_shift'].append(rgb_shift)

        # LAB mean shift
        lab_shift = np.sqrt(
            (stats_current['lab_L_mean'] - ref_stats['lab_L_mean'])**2 +
            (stats_current['lab_A_mean'] - ref_stats['lab_A_mean'])**2 +
            (stats_current['lab_B_mean'] - ref_stats['lab_B_mean'])**2
        )
        results['lab_mean_shift'].append(lab_shift)

        # Brightness shift
        brightness_shift = abs(stats_current['overall_brightness'] - ref_stats['overall_brightness'])
        results['brightness_shift'].append(brightness_shift)

    return results


def plot_color_analysis(all_results: Dict, output_dir: str, style_range: Tuple[int, int]):
    """
    Create comprehensive visualization of color analysis.

    Args:
        all_results: Dict mapping seed -> analysis results
        output_dir: Directory to save plots
        style_range: Tuple of (start, end) style block indices
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # 1. Delta E vs Alpha (perceptual color difference)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Delta E
    ax = axes[0, 0]
    for seed, results in all_results.items():
        ax.plot(results['alphas'], results['delta_e'], marker='o', label=f'Seed {seed}', alpha=0.7)
    ax.set_xlabel('Alpha', fontsize=12)
    ax.set_ylabel('Delta E (Perceptual Color Difference)', fontsize=12)
    ax.set_title('Perceptual Color Difference vs Alpha', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: RGB Mean Shift
    ax = axes[0, 1]
    for seed, results in all_results.items():
        ax.plot(results['alphas'], results['rgb_mean_shift'], marker='s', label=f'Seed {seed}', alpha=0.7)
    ax.set_xlabel('Alpha', fontsize=12)
    ax.set_ylabel('RGB Mean Shift (Euclidean)', fontsize=12)
    ax.set_title('RGB Color Mean Shift vs Alpha', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: LAB Mean Shift
    ax = axes[1, 0]
    for seed, results in all_results.items():
        ax.plot(results['alphas'], results['lab_mean_shift'], marker='^', label=f'Seed {seed}', alpha=0.7)
    ax.set_xlabel('Alpha', fontsize=12)
    ax.set_ylabel('LAB Mean Shift', fontsize=12)
    ax.set_title('LAB Color Space Shift vs Alpha', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Brightness Shift
    ax = axes[1, 1]
    for seed, results in all_results.items():
        ax.plot(results['alphas'], results['brightness_shift'], marker='d', label=f'Seed {seed}', alpha=0.7)
    ax.set_xlabel('Alpha', fontsize=12)
    ax.set_ylabel('Brightness Shift', fontsize=12)
    ax.set_title('Overall Brightness Change vs Alpha', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Color Analysis - Style Blocks {style_range[0]}-{style_range[1]}', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / f'color_analysis_overview_styles{style_range[0]}-{style_range[1]}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Per-channel RGB statistics
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    channels = ['R', 'G', 'B']
    for idx, channel in enumerate(channels):
        # Mean
        ax = axes[0, idx]
        for seed, results in all_results.items():
            means = [stats[f'rgb_{channel}_mean'] for stats in results['color_stats']]
            ax.plot(results['alphas'], means, marker='o', label=f'Seed {seed}', alpha=0.7)
        ax.set_xlabel('Alpha', fontsize=11)
        ax.set_ylabel(f'{channel} Channel Mean', fontsize=11)
        ax.set_title(f'{channel} Channel Mean vs Alpha', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Std
        ax = axes[1, idx]
        for seed, results in all_results.items():
            stds = [stats[f'rgb_{channel}_std'] for stats in results['color_stats']]
            ax.plot(results['alphas'], stds, marker='s', label=f'Seed {seed}', alpha=0.7)
        ax.set_xlabel('Alpha', fontsize=11)
        ax.set_ylabel(f'{channel} Channel Std Dev', fontsize=11)
        ax.set_title(f'{channel} Channel Variation vs Alpha', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'RGB Channel Statistics - Style Blocks {style_range[0]}-{style_range[1]}', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / f'rgb_channel_analysis_styles{style_range[0]}-{style_range[1]}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. LAB channel analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    lab_channels = ['L', 'A', 'B']
    lab_names = ['Lightness', 'Green-Red', 'Blue-Yellow']

    for idx, (channel, name) in enumerate(zip(lab_channels, lab_names)):
        ax = axes[idx]
        for seed, results in all_results.items():
            means = [stats[f'lab_{channel}_mean'] for stats in results['color_stats']]
            ax.plot(results['alphas'], means, marker='o', label=f'Seed {seed}', alpha=0.7)
        ax.set_xlabel('Alpha', fontsize=11)
        ax.set_ylabel(f'{channel} Channel Mean', fontsize=11)
        ax.set_title(f'LAB {name} vs Alpha', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'LAB Color Space Analysis - Style Blocks {style_range[0]}-{style_range[1]}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f'lab_analysis_styles{style_range[0]}-{style_range[1]}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\nPlots saved to {output_dir}/')


def print_summary_statistics(all_results: Dict):
    """Print summary statistics about color shifts."""
    print("\n" + "="*70)
    print("COLOR SHIFT SUMMARY STATISTICS")
    print("="*70)

    for seed, results in all_results.items():
        print(f"\nSeed {seed}:")
        print(f"  Alpha range: {min(results['alphas'])} to {max(results['alphas'])}")
        print(f"  Max Delta E: {max(results['delta_e']):.2f} (perceptual difference)")
        print(f"  Max RGB shift: {max(results['rgb_mean_shift']):.2f}")
        print(f"  Max LAB shift: {max(results['lab_mean_shift']):.2f}")
        print(f"  Max brightness shift: {max(results['brightness_shift']):.2f}")

        # Check for linear trends
        alphas = np.array(results['alphas'])
        delta_e = np.array(results['delta_e'])

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(alphas, delta_e)
        print(f"  Delta E vs Alpha correlation: R² = {r_value**2:.3f}")
        if r_value**2 > 0.8:
            print(f"    → Strong linear relationship (slope={slope:.3f})")

    print("\n" + "="*70)
    print("\nInterpretation Guide:")
    print("  - Delta E < 1.0: Not perceptible by human eyes")
    print("  - Delta E 1-2: Perceptible through close observation")
    print("  - Delta E 2-10: Perceptible at a glance")
    print("  - Delta E > 10: Colors appear quite different")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze color shifts in generated images')
    parser.add_argument('--image-dir', type=str, required=True,
                        help='Directory containing generated images')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save analysis plots (default: image-dir/analysis)')
    parser.add_argument('--reference-alpha', type=int, default=0,
                        help='Alpha value to use as reference (default: 0)')

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.image_dir, 'analysis')

    print(f"Loading images from: {args.image_dir}")
    images = load_images_by_seed_alpha(args.image_dir)

    if not images:
        print("Error: No images found matching the pattern 'seed*_alpha*_styles*.png'")
        return

    print(f"Found {len(images)} seeds with {len(next(iter(images.values())))} alpha values each")

    # Get style range from first image
    first_seed = next(iter(images.keys()))
    first_alpha = next(iter(images[first_seed].keys()))
    style_range = images[first_seed][first_alpha]['style_range']
    print(f"Style range: {style_range[0]}-{style_range[1]}")

    # Analyze each seed
    print("\nAnalyzing color statistics...")
    all_results = {}
    for seed, seed_data in images.items():
        print(f"  Processing seed {seed}...")
        all_results[seed] = analyze_seed_images(seed_data, args.reference_alpha)

    # Print summary
    print_summary_statistics(all_results)

    # Create visualizations
    print(f"\nGenerating plots...")
    plot_color_analysis(all_results, args.output_dir, style_range)

    print(f"\nAnalysis complete! Check {args.output_dir}/ for visualizations.")


if __name__ == '__main__':
    main()


#  Basic usage
# uv run python analyze_color_shift.py --image-dir out/

#  Specify output directory and reference alpha
# uv run python analyze_color_shift.py --image-dir out/ --output-dir analysis_results/ --reference-alpha 0