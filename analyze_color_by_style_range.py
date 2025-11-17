"""
Analyze and compare color shifts across different style ranges.

This script loads images generated with different style ranges and alpha values,
then creates comparison plots showing how each style range affects color metrics.
"""

import os
import re
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from skimage import color
import seaborn as sns


def load_images_by_style_range(image_dir: str, pattern: str = "seed*_alpha*_styles*.png") -> dict:
    """
    Load all generated images and organize by style range, seed, and alpha.

    Returns:
        Dict with structure: {style_range: {seed: {alpha: image_array}}}
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
            style_range = (style_start, style_end)

            # Load image
            img = np.array(Image.open(img_path))

            if style_range not in images:
                images[style_range] = {}
            if seed not in images[style_range]:
                images[style_range][seed] = {}
            images[style_range][seed][alpha] = {
                'image': img,
                'path': str(img_path)
            }

    return images


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to LAB color space."""
    img_normalized = image.astype(np.float32) / 255.0
    lab = color.rgb2lab(img_normalized)
    return lab


def compute_color_statistics(image: np.ndarray) -> dict:
    """Compute comprehensive color statistics for an image."""
    stats_dict = {}

    # RGB statistics
    for i, channel in enumerate(['R', 'G', 'B']):
        channel_data = image[:, :, i].flatten()
        stats_dict[f'rgb_{channel}_mean'] = np.mean(channel_data)
        stats_dict[f'rgb_{channel}_std'] = np.std(channel_data)

    # LAB statistics
    lab = rgb_to_lab(image)
    for i, channel in enumerate(['L', 'A', 'B']):
        channel_data = lab[:, :, i].flatten()
        stats_dict[f'lab_{channel}_mean'] = np.mean(channel_data)
        stats_dict[f'lab_{channel}_std'] = np.std(channel_data)

    # Overall statistics
    stats_dict['overall_brightness'] = np.mean(image)

    return stats_dict


def compute_delta_e(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute average Delta E (CIE76) between two images."""
    lab1 = rgb_to_lab(img1)
    lab2 = rgb_to_lab(img2)
    delta_e = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=2))
    return np.mean(delta_e)


def analyze_style_range_data(style_range_data: dict, reference_alpha: int = 0) -> dict:
    """
    Analyze all images for a single style range across seeds and alphas.

    Args:
        style_range_data: Dict mapping seed -> {alpha -> image data}
        reference_alpha: Alpha value to use as reference

    Returns:
        Dict with analysis results aggregated across seeds
    """
    # Get all alphas (should be same for all seeds)
    first_seed = next(iter(style_range_data.keys()))
    alphas = sorted(style_range_data[first_seed].keys())

    # Find reference alpha
    if reference_alpha not in alphas:
        reference_alpha = min(alphas, key=lambda x: abs(x))

    results = {
        'alphas': alphas,
        'delta_e': [],
        'rgb_mean_shift': [],
        'lab_mean_shift': [],
        'brightness_shift': [],
        'rgb_R_mean': [],
        'rgb_G_mean': [],
        'rgb_B_mean': [],
        'rgb_R_std': [],
        'rgb_G_std': [],
        'rgb_B_std': [],
        'lab_L_mean': [],
        'lab_A_mean': [],
        'lab_B_mean': [],
    }

    # For each alpha, aggregate across all seeds
    for alpha in alphas:
        alpha_delta_e = []
        alpha_rgb_shift = []
        alpha_lab_shift = []
        alpha_brightness_shift = []
        alpha_stats = {key: [] for key in ['rgb_R_mean', 'rgb_G_mean', 'rgb_B_mean',
                                             'rgb_R_std', 'rgb_G_std', 'rgb_B_std',
                                             'lab_L_mean', 'lab_A_mean', 'lab_B_mean']}

        for seed, seed_data in style_range_data.items():
            img = seed_data[alpha]['image']
            ref_img = seed_data[reference_alpha]['image']

            # Compute metrics
            stats_current = compute_color_statistics(img)
            stats_ref = compute_color_statistics(ref_img)

            # Delta E
            delta_e = compute_delta_e(ref_img, img)
            alpha_delta_e.append(delta_e)

            # RGB mean shift
            rgb_shift = np.sqrt(
                (stats_current['rgb_R_mean'] - stats_ref['rgb_R_mean'])**2 +
                (stats_current['rgb_G_mean'] - stats_ref['rgb_G_mean'])**2 +
                (stats_current['rgb_B_mean'] - stats_ref['rgb_B_mean'])**2
            )
            alpha_rgb_shift.append(rgb_shift)

            # LAB mean shift
            lab_shift = np.sqrt(
                (stats_current['lab_L_mean'] - stats_ref['lab_L_mean'])**2 +
                (stats_current['lab_A_mean'] - stats_ref['lab_A_mean'])**2 +
                (stats_current['lab_B_mean'] - stats_ref['lab_B_mean'])**2
            )
            alpha_lab_shift.append(lab_shift)

            # Brightness shift
            brightness_shift = abs(stats_current['overall_brightness'] - stats_ref['overall_brightness'])
            alpha_brightness_shift.append(brightness_shift)

            # Collect stats
            for key in alpha_stats.keys():
                alpha_stats[key].append(stats_current[key])

        # Average across seeds
        results['delta_e'].append(np.mean(alpha_delta_e))
        results['rgb_mean_shift'].append(np.mean(alpha_rgb_shift))
        results['lab_mean_shift'].append(np.mean(alpha_lab_shift))
        results['brightness_shift'].append(np.mean(alpha_brightness_shift))

        for key in alpha_stats.keys():
            results[key].append(np.mean(alpha_stats[key]))

    return results


def get_style_range_label(style_range: tuple[int, int]) -> str:
    """Get a descriptive label for a style range."""
    start, end = style_range
    if start <= 8 and end <= 8:
        return f"Coarse ({start}-{end})"
    elif start >= 4 and end <= 12:
        return f"Middle ({start}-{end})"
    elif start >= 12:
        return f"Fine ({start}-{end})"
    else:
        return f"Blocks {start}-{end}"


def plot_comparison_by_style_range(all_results: dict, output_dir: str):
    """
    Create comparison plots showing different style ranges.

    Args:
        all_results: Dict mapping style_range -> analysis results
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # Color palette for different style ranges
    style_colors = {}
    color_palette = sns.color_palette("husl", len(all_results))
    for idx, style_range in enumerate(sorted(all_results.keys())):
        style_colors[style_range] = color_palette[idx]

    # Get alphas (should be same for all style ranges)
    alphas = next(iter(all_results.values()))['alphas']
    ref_idx = min(range(len(alphas)), key=lambda i: abs(alphas[i]))

    # 1. Overview plot with 4 key metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Delta E
    ax = axes[0, 0]
    for style_range in sorted(all_results.keys()):
        results = all_results[style_range]
        label = get_style_range_label(style_range)
        ax.plot(results['alphas'], results['delta_e'],
                marker='o', linewidth=2, label=label, color=style_colors[style_range])
    ax.set_xlabel('Alpha', fontsize=12)
    ax.set_ylabel('Delta E (Perceptual Color Difference)', fontsize=12)
    ax.set_title('Perceptual Color Difference vs Alpha', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

    # Plot 2: RGB Mean Shift
    ax = axes[0, 1]
    for style_range in sorted(all_results.keys()):
        results = all_results[style_range]
        label = get_style_range_label(style_range)
        ax.plot(results['alphas'], results['rgb_mean_shift'],
                marker='s', linewidth=2, label=label, color=style_colors[style_range])
    ax.set_xlabel('Alpha', fontsize=12)
    ax.set_ylabel('RGB Mean Shift (Euclidean)', fontsize=12)
    ax.set_title('RGB Color Mean Shift vs Alpha', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

    # Plot 3: LAB Mean Shift
    ax = axes[1, 0]
    for style_range in sorted(all_results.keys()):
        results = all_results[style_range]
        label = get_style_range_label(style_range)
        ax.plot(results['alphas'], results['lab_mean_shift'],
                marker='^', linewidth=2, label=label, color=style_colors[style_range])
    ax.set_xlabel('Alpha', fontsize=12)
    ax.set_ylabel('LAB Mean Shift', fontsize=12)
    ax.set_title('LAB Color Space Shift vs Alpha', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

    # Plot 4: Brightness Shift
    ax = axes[1, 1]
    for style_range in sorted(all_results.keys()):
        results = all_results[style_range]
        label = get_style_range_label(style_range)
        ax.plot(results['alphas'], results['brightness_shift'],
                marker='d', linewidth=2, label=label, color=style_colors[style_range])
    ax.set_xlabel('Alpha', fontsize=12)
    ax.set_ylabel('Brightness Shift', fontsize=12)
    ax.set_title('Overall Brightness Change vs Alpha', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

    plt.suptitle('Color Analysis by Style Range (Averaged Across Seeds)',
                 fontsize=16, y=0.995, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'color_analysis_by_style_range_overview.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. RGB Channel Analysis - individual channels as % change + red proportion
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    rgb_channels = ['R', 'G', 'B']
    channel_colors = {'R': '#E63946', 'G': '#06D6A0', 'B': '#118AB2'}

    # Columns 0-2: Mean plots for each channel
    for col_idx, channel in enumerate(rgb_channels):
        ax = axes[col_idx]
        for style_range in sorted(all_results.keys()):
            results = all_results[style_range]
            label = get_style_range_label(style_range)
            vals = np.array(results[f'rgb_{channel}_mean'])
            ref_val = vals[ref_idx]
            pct_change = (vals / ref_val - 1) * 100
            ax.plot(results['alphas'], pct_change,
                    marker='o', linewidth=2, label=label, color=style_colors[style_range])

        ax.set_xlabel('Alpha', fontsize=11)
        ax.set_ylabel(f'{channel} Channel Mean Change (%)', fontsize=11)
        ax.set_title(f'{channel} Channel Mean vs Alpha', fontsize=12, fontweight='bold', color=channel_colors[channel])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

    # Column 3: Red proportion (R / (R+G+B))
    ax = axes[3]
    for style_range in sorted(all_results.keys()):
        results = all_results[style_range]
        label = get_style_range_label(style_range)

        r_vals = np.array(results['rgb_R_mean'])
        g_vals = np.array(results['rgb_G_mean'])
        b_vals = np.array(results['rgb_B_mean'])

        # Calculate red proportion
        red_proportion = r_vals / (r_vals + g_vals + b_vals)
        ref_red_proportion = red_proportion[ref_idx]

        # Convert to percentage change
        pct_change = (red_proportion / ref_red_proportion - 1) * 100

        ax.plot(results['alphas'], pct_change,
                marker='D', linewidth=2, label=label, color=style_colors[style_range])

    ax.set_xlabel('Alpha', fontsize=11)
    ax.set_ylabel('Red Proportion Change (%)', fontsize=11)
    ax.set_title('Red Proportion vs Alpha\n(R / (R+G+B))', fontsize=12, fontweight='bold', color='#E63946')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

    plt.suptitle('RGB Channel Analysis by Style Range (% Change)',
                 fontsize=16, y=0.995, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'rgb_analysis_by_style_range.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. LAB channel analysis - individual channels as % change
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    lab_channels = ['L', 'A', 'B']
    lab_names = ['Lightness (L)', 'Green-Red (a*)', 'Blue-Yellow (b*)']
    lab_colors = {'L': '#FFB703', 'A': '#06D6A0', 'B': '#118AB2'}

    for col_idx, (channel, name) in enumerate(zip(lab_channels, lab_names)):
        ax = axes[col_idx]
        for style_range in sorted(all_results.keys()):
            results = all_results[style_range]
            label = get_style_range_label(style_range)
            vals = np.array(results[f'lab_{channel}_mean'])
            ref_val = vals[ref_idx]
            pct_change = (vals / ref_val - 1) * 100
            ax.plot(results['alphas'], pct_change,
                    marker='o', linewidth=2, label=label, color=style_colors[style_range])

        ax.set_xlabel('Alpha', fontsize=11)
        ax.set_ylabel(f'{name} Change (%)', fontsize=11)
        ax.set_title(f'{name} vs Alpha', fontsize=12, fontweight='bold', color=lab_colors[channel])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

    plt.suptitle('LAB Color Space by Style Range (% Change)',
                 fontsize=16, y=0.995, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'lab_analysis_by_style_range.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\nComparison plots saved to {output_dir}/')


def print_summary_statistics(all_results: dict):
    """Print summary statistics about color shifts for each style range."""
    print("\n" + "="*70)
    print("COLOR SHIFT COMPARISON BY STYLE RANGE")
    print("="*70)

    for style_range in sorted(all_results.keys()):
        results = all_results[style_range]
        label = get_style_range_label(style_range)

        print(f"\n{label}:")
        print(f"  Alpha range: {min(results['alphas'])} to {max(results['alphas'])}")
        print(f"  Max Delta E: {max(results['delta_e']):.2f}")
        print(f"  Max RGB shift: {max(results['rgb_mean_shift']):.2f}")
        print(f"  Max LAB shift: {max(results['lab_mean_shift']):.2f}")
        print(f"  Max brightness shift: {max(results['brightness_shift']):.2f}")

        # Linear correlation
        alphas_arr = np.array(results['alphas'])
        delta_e_arr = np.array(results['delta_e'])
        slope, intercept, r_value, p_value, std_err = stats.linregress(alphas_arr, delta_e_arr)
        print(f"  Delta E vs Alpha correlation: R² = {r_value**2:.3f}")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Compare color shifts across different style ranges')
    parser.add_argument('--image-dir', type=str, required=True,
                        help='Directory containing generated images')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save analysis plots (default: image-dir/analysis_by_range)')
    parser.add_argument('--reference-alpha', type=int, default=0,
                        help='Alpha value to use as reference (default: 0)')

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.image_dir, 'analysis_by_range')

    print(f"Loading images from: {args.image_dir}")
    images = load_images_by_style_range(args.image_dir)

    if not images:
        print("Error: No images found matching the pattern 'seed*_alpha*_styles*.png'")
        return

    print(f"Found {len(images)} style ranges:")
    for style_range in sorted(images.keys()):
        num_seeds = len(images[style_range])
        num_alphas = len(next(iter(images[style_range].values())))
        print(f"  {get_style_range_label(style_range)}: {num_seeds} seeds × {num_alphas} alphas")

    # Analyze each style range
    print("\nAnalyzing color statistics by style range...")
    all_results = {}
    for style_range, style_data in images.items():
        print(f"  Processing {get_style_range_label(style_range)}...")
        all_results[style_range] = analyze_style_range_data(style_data, args.reference_alpha)

    # Print summary
    print_summary_statistics(all_results)

    # Create visualizations
    print(f"\nGenerating comparison plots...")
    plot_comparison_by_style_range(all_results, args.output_dir)

    print(f"\nAnalysis complete! Check {args.output_dir}/ for visualizations.")


if __name__ == '__main__':
    main()


#  Basic usage
# uv run python analyze_color_by_style_range.py --image-dir out/

#  Specify output directory and reference alpha
# uv run python analyze_color_by_style_range.py --image-dir out/ --output-dir analysis_results/ --reference-alpha 0
