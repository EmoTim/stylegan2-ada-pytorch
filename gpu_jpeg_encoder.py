import torch
from nvidia import nvimgcodec
import numpy as np

class GPUJPEGEncoder:
    def __init__(self, quality=95):
        self.encoder = nvimgcodec.Encoder()
        self.params = nvimgcodec.EncodeParams(quality=quality)

    def encode_batch(self, batch_tensor):
        """
        batch_tensor: torch.Tensor [B, H, W, 3], uint8, CUDA (channel-last)

        Returns: list of JPEG bytes for each image
        """
        assert batch_tensor.is_cuda
        assert batch_tensor.dtype == torch.uint8
        B, H, W, C = batch_tensor.shape
        assert C == 3, "Expecting RGB images"

        # Ensure contiguous memory layout
        batch_chlast = batch_tensor.contiguous()

        # Wrap into nvimgcodec Images (GPU tensor is passed directly)
        imgs = nvimgcodec.as_images(batch_chlast)

        # GPU-side JPEG encoding (batched)
        encoded = self.encoder.encode(imgs, self.params)

        # Convert each encoded image to raw bytes
        jpeg_list = [enc.cpu().numpy().tobytes() for enc in encoded]
        return jpeg_list
