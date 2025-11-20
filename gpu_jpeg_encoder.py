import torch
from nvidia import nvimgcodec
import numpy as np

class GPUJPEGEncoder:
    def __init__(self, quality=95):
        self.encoder = nvimgcodec.Encoder()
        self.params = nvimgcodec.EncodeParams(quality=quality)

    def encode_batch(self, batch_tensor):
        """
        batch_tensor: torch.Tensor [B, 3, H, W], uint8, CUDA
        
        Returns: list of JPEG bytes for each image
        """
        assert batch_tensor.is_cuda
        assert batch_tensor.dtype == torch.uint8
        B, C, H, W = batch_tensor.shape
        assert C == 3, "Expecting RGB images"

        # Convert to (B, H, W, 3) channel-last format for nvimgcodec
        batch_chlast = batch_tensor.permute(0, 2, 3, 1).contiguous()

        # Wrap into nvimgcodec Images (GPU tensor is passed directly)
        imgs = nvimgcodec.as_images(batch_chlast)

        # GPU-side JPEG encoding (batched)
        encoded = self.encoder.encode(imgs, self.params)

        # Convert each encoded image to raw bytes
        jpeg_list = [enc.cpu().numpy().tobytes() for enc in encoded]
        return jpeg_list
