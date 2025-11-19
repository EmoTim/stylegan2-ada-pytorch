import insightface
from insightface.app import FaceAnalysis
import numpy as np
from PIL import Image


class AgeEstimator:
    def __init__(
        self, ctx_id: int = -1, det_size: tuple[int, int] = (640, 640)
    ) -> None:
        self.app = FaceAnalysis(name="buffalo_l")  # SOTA Model including age
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)  # -1 = CPU, 0+ = GPU

    def estimate_age(self, pil_img: Image.Image) -> float | None:
        """
        Estimate age from a PIL Image.

        Args:
            pil_img: PIL Image object

        Returns:
            Estimated age as float, or None if no face detected
        """
        img = np.array(pil_img)
        img = img[:, :, ::-1]  # Convert RGB to BGR for InsightFace
        faces = self.app.get(img)
        if len(faces) == 0:
            return None
        return faces[0].age

    def __call__(self, pil_img: Image.Image) -> float | None:
        """Allow instance to be called directly."""
        return self.estimate_age(pil_img)
