import insightface
from insightface.app import FaceAnalysis
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F 
from vgg import VGG


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


class AgePredictor:
    def __init__(self):
        self.age_net = VGG()
        ckpt = torch.load("dex_age_classifier.pth", map_location="cpu")['state_dict']
        ckpt = {k.replace('-', '_'): v for k, v in ckpt.items()}
        self.age_net.load_state_dict(ckpt)
        self.age_net.cuda()
        self.age_net.eval()
        self.min_age = 0
        self.max_age = 100

    def __get_predicted_age(self, age_pb):
        predict_age_pb = F.softmax(age_pb)
        predict_age = torch.zeros(age_pb.size(0)).type_as(predict_age_pb)
        for i in range(age_pb.size(0)):
            for j in range(age_pb.size(1)):
                predict_age[i] += j * predict_age_pb[i][j]
        return predict_age

    def extract_ages(self, x):
        x = torch.from_numpy(np.array(x))
        # Convert from uint8 to float and normalize to [0, 1]
        x = x.float() / 255.0
        # permute to (C, H, W)
        x = x.permute(2, 0, 1)  # shape: [3, 1024, 1024]
        # add batch dimension
        x = x.unsqueeze(0)  # shape: [1, 3, 1024, 1024]
        # Move to GPU
        x = x.cuda()
        # interpolate
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)  # shape: [1, 3, 224, 224]
        predict_age_pb = self.age_net(x)['fc8']
        predicted_age = self.__get_predicted_age(predict_age_pb)
        return predicted_age
    
    def __call__(self, pil_img: Image.Image) -> float | None:
        """Allow instance to be called directly."""
        return self.extract_ages(pil_img)[0]
    