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
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--weight-vector', help='Path to weight vector file (.npy)', type=str, metavar='FILE')
@click.option('--alphas', type=num_range, help='Alpha values for weight modulation (e.g., "-10,0,10" or "-10-10")', default='-10,0,10', show_default=True)
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str],
    weight_vector: Optional[str],
    alphas: List[int]
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
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print ('warn: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
        return

    if seeds is None:
        ctx.fail('--seeds option is required when not using --projected-w')

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # Load weight vector if specified.
    weight_vec = None
    if weight_vector is not None:
        print(f'Loading weight vector from "{weight_vector}"')
        weight_vec = torch.tensor(np.load(weight_vector), device=device)
        print(f'Weight vector shape: {weight_vec.shape}')

    # Generate images.
    all_images = []  # Store images for composite: list of lists (one per seed)

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        w = G.mapping(z, label)

        if weight_vec is not None:
            # Generate images with weight modulation
            seed_images = []
            for alpha in alphas:
                w_modified = w + alpha * weight_vec.unsqueeze(0)
                assert w_modified.shape[1:] == (G.num_ws, G.w_dim)
                img = G.synthesis(w_modified, noise_mode=noise_mode)
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img_array = img[0].cpu().numpy()
                pil_img = PIL.Image.fromarray(img_array, 'RGB')
                pil_img.save(f'{outdir}/seed{seed:04d}_alpha{alpha}.png')
                seed_images.append(img_array)
            all_images.append(seed_images)
        else:
            # Generate image without weight modulation
            img = G.synthesis(w, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

    # Create composite image if weight modulation was used
    if weight_vec is not None and len(all_images) > 0:
        print('Creating composite image...')
        num_rows = len(all_images)
        num_cols = len(alphas)
        img_height, img_width = all_images[0][0].shape[:2]

        # Create composite canvas
        composite = np.zeros((num_rows * img_height, num_cols * img_width, 3), dtype=np.uint8)

        for row_idx, seed_images in enumerate(all_images):
            for col_idx, img_array in enumerate(seed_images):
                y_start = row_idx * img_height
                y_end = (row_idx + 1) * img_height
                x_start = col_idx * img_width
                x_end = (col_idx + 1) * img_width
                composite[y_start:y_end, x_start:x_end] = img_array

        composite_img = PIL.Image.fromarray(composite, 'RGB')
        composite_path = f'{outdir}/composite_grid.png'
        composite_img.save(composite_path)
        print(f'Composite image saved to {composite_path}')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Default arguments for debugging/development
    if len(sys.argv) == 1:
        sys.argv = [
            'generate.py',
            '--outdir=out',
            '--trunc=0.7',
            '--seeds=600-605',
            '--weight-vector=weight.npy',
            '--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl'
        ]

    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
