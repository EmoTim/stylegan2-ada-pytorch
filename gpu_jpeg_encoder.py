import torch
from nvidia import nvimgcodec
import numpy as np

class GPUJPEGEncoder:
    def __init__(self, quality=95):
        self.encoder = nvimgcodec.Encoder()
        self.params = nvimgcodec.EncodeParams(
            quality_type=nvimgcodec.QualityType.DEFAULT,
            quality_value=float(quality)
        )

    def encode_batch(self, batch_tensor):
        """
        batch_tensor: torch.Tensor [B, H, W, 3], uint8, CUDA (channel-last)

        Returns: list of JPEG bytes for each image
        """
        assert batch_tensor.is_cuda
        assert batch_tensor.dtype == torch.uint8
        B, H, W, C = batch_tensor.shape
        assert C == 3, "Expecting RGB images"


        jpeg_list = []
        for img in batch_tensor:
            img_chlast = img.contiguous()  # [H,W,3]
            nv_img = nvimgcodec.as_images([img_chlast])[0]   # single image
            encoded = self.encoder.encode(nv_img, codec="jpeg", params=self.params)
            jpeg_list.append(encoded)
        return jpeg_list
