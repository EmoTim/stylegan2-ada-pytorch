# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
import warnings
from typing import List, Optional
import io
import time

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import matplotlib.pyplot as plt
import boto3

import legacy
from age_estimator import AgePredictor

# Suppress CUDA kernel compilation warnings
warnings.filterwarnings('ignore', message='Failed to build CUDA kernels for upfirdn2d')

# ----------------------------------------------------------------------------


def float_range(s: str) -> List[float]:

    parts = s.split(":")
    if len(parts) == 3:
        start, stop, num = float(parts[0]), float(parts[1]), int(parts[2])
        return list(np.round(np.linspace(start, stop, num), decimals=1))
    else:
        raise ValueError(
            f"Linspace format requires exactly 3 values (start:stop:num), got {len(parts)}"
        )
    
def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


# ----------------------------------------------------------------------------


@click.command()
@click.pass_context
@click.option("--network", "network_pkl", help="Network pickle filename", required=True)
@click.option(
    "--vgg-path",
    help="Path to vgg age predictor",
    type=str,
    metavar="FILE",
)
@click.option("--seeds", type=num_range, help="List of random seeds")
@click.option(
    "--trunc",
    "truncation_psi",
    type=float,
    help="Truncation psi",
    default=1,
    show_default=True,
)
@click.option(
    "--class",
    "class_idx",
    type=int,
    help="Class label (unconditional if not specified)",
)
@click.option(
    "--noise-mode",
    help="Noise mode",
    type=click.Choice(["const", "random", "none"]),
    default="const",
    show_default=True,
)
@click.option("--projected-w", help="Projection result file", type=str, metavar="FILE")
@click.option(
    "--outdir",
    help="Where to save the output images",
    type=str,
    required=True,
    metavar="DIR",
)
@click.option(
    "--weight-vector",
    help="Path to weight vector file (.npy)",
    type=str,
    metavar="FILE",
)

@click.option(
    "--alphas",
    type=float_range,
    help='Alpha values for weight modulation (e.g., "-10,0,10" or "-10-10")',
    default="-5, -2, -1, 0, 1, 2, 5",
    show_default=True,
)
@click.option(
    "--style-range",
    type=(int, int),
    help="Range of style blocks to apply weight vector (start, end). Coarse: 0-8, Middle: 4-12, Fine: 12-17. Default: all (0, 17)",
    default=(0, 17),
    show_default=True,
)
@click.option(
    "--create-composite",
    type=bool,
    help="Bolean to create or not a composite grid of the generated images",
    default=False,
    show_default=True,
)
@click.option(
    "--s3-bucket",
    help="S3 bucket to upload images to (e.g., 's3://bucket-name/path/to/images')",
    type=str,
    metavar="S3_URI",
)
@click.option(
    "--batch-size",
    help="Number of seeds to process in parallel (default: 1)",
    type=int,
    default=1,
    show_default=True,
)

def generate_images(
    ctx: click.Context,
    network_pkl: str,
    vgg_path: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str],
    weight_vector: Optional[str],
    alphas: List[int],
    style_range: tuple,
    create_composite: bool = False,
    s3_bucket: Optional[str] = None,
    batch_size: int = 1,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    
    \b
    # (Tim) Generate FFHQ images with weight modulation
    python generate.py --outdir=stylegan2-ada-pytorch/out --trunc=0.7 --seeds=600-605 \\
        --weight-vector=weight.npy --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

    \b
    # (Tim) Generate with weight modulation on coarse styles only (pose, face shape, eyeglasses)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --weight-vector=weight.npy --style-range 0 8 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

    \b
    # (Tim) Generate with weight modulation on middle styles (facial features, hair style, eyes)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --weight-vector=weight.npy --style-range 4 12 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

    \b
    # (Tim) Generate with weight modulation on fine styles (color scheme, microstructure)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --weight-vector=weight.npy --style-range 12 17 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device("cuda")
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Initialize S3 client if bucket is specified
    s3_client = None
    s3_bucket_name = None
    s3_prefix = None
    if s3_bucket:
        # Parse S3 URI (e.g., s3://bucket-name/path/to/images)
        if s3_bucket.startswith('s3://'):
            s3_bucket = s3_bucket[5:]  # Remove 's3://'
        parts = s3_bucket.split('/', 1)
        s3_bucket_name = parts[0]
        s3_prefix = parts[1] if len(parts) > 1 else ''
        if s3_prefix and not s3_prefix.endswith('/'):
            s3_prefix += '/'

        s3_client = boto3.client('s3')
        print(f'S3 upload enabled: s3://{s3_bucket_name}/{s3_prefix}')

    def save_image(pil_img, file_path):
        """Save image locally and optionally upload to S3."""
        # Save locally
        pil_img.save(file_path)

        # Upload to S3 if enabled
        if s3_client:
            s3_key = s3_prefix + file_path.replace(outdir + '/', '')
            try:
                # Upload directly from memory buffer
                img_buffer = io.BytesIO()
                pil_img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                s3_client.upload_fileobj(img_buffer, s3_bucket_name, s3_key)
                print(f'  ✓ Uploaded to s3://{s3_bucket_name}/{s3_key}')
            except Exception as e:
                print(f'  ⚠️ S3 upload failed for {s3_key}: {e}')

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print("warn: --seeds is ignored when using --projected-w")
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)["w"]
        ws = torch.tensor(ws, device=device)  # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            pil_img = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")
            save_image(pil_img, f"{outdir}/proj{idx:02d}.png")
        return

    if seeds is None:
        ctx.fail("--seeds option is required when not using --projected-w")

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail(
                "Must specify class label with --class when using a conditional network"
            )
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print("warn: --class=lbl ignored when running on an unconditional network")

    # Load weight vector if specified.
    weight_vec = None
    if weight_vector is not None:
        print(f'Loading weight vector from "{weight_vector}"')
        weight_vec = torch.tensor(np.load(weight_vector), device=device)
        print(f"Weight vector shape: {weight_vec.shape}")

        # Validate style range
        start_idx, end_idx = style_range
        if not (0 <= start_idx <= 17 and 0 <= end_idx <= 17):
            ctx.fail("Style range indices must be between 0 and 17")
        if start_idx > end_idx:
            ctx.fail("Style range start must be <= end")
        print(
            f"Applying weight vector to style blocks {start_idx} to {end_idx} (inclusive)"
        )

    # Initialize age predictor if using weight vector
    age_predictor = None
    if weight_vector is not None:
        if vgg_path is None:
            ctx.fail("--vgg-path is required when using --weight-vector for age filtering")
        print("Initializing age predictor...")
        age_predictor = AgePredictor(vgg_path=vgg_path)

    # Generate images.
    all_images = []  # Store images for composite: list of lists (one per seed)

    total_start_time = time.time()
    total_images_generated = 0

    # Process seeds in batches
    for batch_start in range(0, len(seeds), batch_size):
        batch_end = min(batch_start + batch_size, len(seeds))
        batch_seeds = seeds[batch_start:batch_end]
        current_batch_size = len(batch_seeds)

        batch_num = batch_start//batch_size + 1
        total_batches = (len(seeds) + batch_size - 1)//batch_size
        print(f"Processing batch {batch_num}/{total_batches} (seeds {batch_start}-{batch_end-1})...")
        batch_start_time = time.time()

        # Generate latent codes for all seeds in batch
        z_batch = torch.stack([
            torch.from_numpy(np.random.RandomState(seed).randn(G.z_dim))
            for seed in batch_seeds
        ]).to(device)

        # Expand label to match batch size
        label_batch = label.repeat(current_batch_size, 1)

        # Generate w latent codes for batch
        w_batch = G.mapping(z_batch, label_batch, truncation_psi=truncation_psi)

        if weight_vec is not None:
            start_idx, end_idx = style_range

            # First, check ages with alpha=0 if age estimator is available
            valid_indices = []
            if age_predictor is not None:
                age_check_start = time.time()
                imgs = G.synthesis(w_batch, noise_mode=noise_mode)
                age_check_synthesis_time = time.time() - age_check_start

                for i, seed in enumerate(batch_seeds):
                    estimated_age = age_predictor(imgs[i:i+1])

                    if estimated_age is None:
                        print(f"  ⚠️  Seed {seed}: No face detected, skipping...")
                    elif estimated_age <= 20:
                        print(f"  ⚠️  Seed {seed}: Age {estimated_age:.1f} <= 20, skipping...")
                    else:
                        print(f"  ✓ Seed {seed}: Age {estimated_age:.1f} > 20, proceeding...")
                        valid_indices.append(i)

                age_check_total_time = time.time() - age_check_start
                print(f"  ⏱️  Age checking: {age_check_total_time:.2f}s (synthesis: {age_check_synthesis_time:.2f}s, prediction: {age_check_total_time - age_check_synthesis_time:.2f}s)")
            else:
                # No age predictor, all seeds are valid
                valid_indices = list(range(current_batch_size))

            # Process valid seeds - batch across all alphas for all valid seeds simultaneously
            if len(valid_indices) > 0:
                # Extract valid w vectors
                valid_w_batch = w_batch[valid_indices]
                valid_seeds = [batch_seeds[i] for i in valid_indices]

                # Initialize storage for each valid seed's images
                seed_images_dict = {seed: [] for seed in valid_seeds}

                # For each alpha, generate images for all valid seeds at once
                alpha_generation_start = time.time()
                for alpha in alphas:
                    # Create modified w for all valid seeds with this alpha
                    w_modified_batch = valid_w_batch.clone()
                    # Apply weight vector to all seeds in batch
                    w_modified_batch[:, start_idx:end_idx + 1, :] += alpha * weight_vec[start_idx:end_idx + 1, :].unsqueeze(0)
                    assert w_modified_batch.shape[1:] == (G.num_ws, G.w_dim)

                    # Generate all images for this alpha in one synthesis call
                    alpha_synthesis_start = time.time()
                    imgs = G.synthesis(w_modified_batch, noise_mode=noise_mode)
                    imgs = (
                        (imgs.permute(0, 2, 3, 1) * 127.5 + 128)
                        .clamp(0, 255)
                        .to(torch.uint8)
                    )
                    alpha_synthesis_time = time.time() - alpha_synthesis_start

                    # Save each image
                    save_start = time.time()
                    for i, seed in enumerate(valid_seeds):
                        img_array = imgs[i].cpu().numpy()
                        pil_img = PIL.Image.fromarray(img_array, "RGB")

                        # Create subdirectory for this alpha value
                        alpha_dir = os.path.join(outdir, f"alpha_{alpha}")
                        os.makedirs(alpha_dir, exist_ok=True)

                        file_path = f"{alpha_dir}/seed{seed:04d}_styles{start_idx}-{end_idx}.png"
                        save_image(pil_img, file_path)

                        # Store for composite
                        seed_images_dict[seed].append(img_array)
                        total_images_generated += 1
                    save_time = time.time() - save_start
                    print(f"  ⏱️  Alpha {alpha:+.1f}: synthesis={alpha_synthesis_time:.2f}s, save={save_time:.2f}s ({len(valid_seeds)} images)")

                alpha_generation_time = time.time() - alpha_generation_start
                print(f"  ⏱️  Total alpha generation: {alpha_generation_time:.2f}s for {len(alphas)} alphas × {len(valid_seeds)} seeds = {len(alphas) * len(valid_seeds)} images")

                # Add all seed images to all_images list
                if create_composite:
                    for seed in valid_seeds:
                        all_images.append(seed_images_dict[seed])
        else:
            # Generate images without weight modulation (batch mode)
            synthesis_start = time.time()
            imgs = G.synthesis(w_batch, truncation_psi=truncation_psi, noise_mode=noise_mode)
            imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            synthesis_time = time.time() - synthesis_start

            save_start = time.time()
            for i, seed in enumerate(batch_seeds):
                pil_img = PIL.Image.fromarray(imgs[i].cpu().numpy(), "RGB")
                save_image(pil_img, f"{outdir}/seed{seed:04d}.png")
                total_images_generated += 1
            save_time = time.time() - save_start
            print(f"  ⏱️  Synthesis: {synthesis_time:.2f}s, Save: {save_time:.2f}s ({current_batch_size} images)")

        batch_time = time.time() - batch_start_time
        print(f"  ⏱️  Batch {batch_num} completed in {batch_time:.2f}s\n")

    # Print final statistics
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print("📊 Generation Statistics:")
    print(f"{'='*60}")
    print(f"  Total images generated: {total_images_generated}")
    print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"  Average time per image: {total_time/total_images_generated:.2f}s")
    print(f"  Throughput: {total_images_generated/total_time:.2f} images/second")
    print(f"  Batch size: {batch_size}")
    print(f"{'='*60}\n")

    # Create composite image if weight modulation was used
    if create_composite and weight_vec is not None and len(all_images) > 0:
        print("Creating composite image...")
        num_rows = len(all_images)
        num_cols = len(alphas)
        start_idx, end_idx = style_range

        # Create matplotlib figure
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3)
        )

        # Handle case where there's only one row or one column
        if num_rows == 1 and num_cols == 1:
            axes = np.array([[axes]])
        elif num_rows == 1:
            axes = axes.reshape(1, -1)
        elif num_cols == 1:
            axes = axes.reshape(-1, 1)

        # Plot images in grid
        for row_idx, seed_images in enumerate(all_images):
            for col_idx, img_array in enumerate(seed_images):
                ax = axes[row_idx, col_idx]
                ax.imshow(img_array)
                ax.axis("off")

                # Add column labels (alpha values) on top row
                if row_idx == 0:
                    ax.set_title(f"α={alphas[col_idx]}", fontsize=12)

                # Add row labels (seed numbers) on left column
                if col_idx == 0:
                    ax.set_ylabel(
                        f"Seed {seeds[row_idx]}",
                        fontsize=12,
                        rotation=0,
                        labelpad=40,
                        va="center",
                    )

        # Add overall title
        style_type = "all styles"
        if start_idx <= 8 and end_idx <= 8:
            style_type = "coarse styles (pose, face shape)"
        elif start_idx >= 4 and end_idx <= 12:
            style_type = "middle styles (facial features)"
        elif start_idx >= 12:
            style_type = "fine styles (color, microstructure)"
        fig.suptitle(
            f"Weight Vector Applied to Blocks {start_idx}-{end_idx} ({style_type})",
            fontsize=14,
            y=1.0,
        )

        plt.tight_layout()
        composite_path = f"{outdir}/composite_grid_styles{start_idx}-{end_idx}.png"
        plt.savefig(composite_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Composite image saved to {composite_path}")

        # Upload composite to S3 if enabled
        if s3_client:
            s3_key = s3_prefix + composite_path.replace(outdir + '/', '')
            try:
                s3_client.upload_file(composite_path, s3_bucket_name, s3_key)
                print(f'  ✓ Composite uploaded to s3://{s3_bucket_name}/{s3_key}')
            except Exception as e:
                print(f'  ⚠️ S3 upload failed for composite: {e}')


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    os.chdir("/home/sagemaker-user/stylegan2-ada-pytorch")

    # Default arguments for debugging/development
    if len(sys.argv) == 1:
        sys.argv = [
            "generate.py",
            "--outdir=out",
            "--trunc=0.7",
            "--seeds=600-605",
            "--style-range", "0", "4",
            "--alphas=0:0:1",
            "--weight-vector=weight.npy",
            "--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
            "--vgg-path=/home/sagemaker-user/stylegan2-ada-pytorch/dex_age_classifier.pth"
        ]

    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
