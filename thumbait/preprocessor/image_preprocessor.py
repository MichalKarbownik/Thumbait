import torch

import torchvision.transforms as T
import numpy as np


class ImagePreprocessor:
    def __init__(self):
        self.data_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.Resize(size=(180, 240)),
            ]
        )

    def transform_image(self, image):
        return self.data_transform(image)

    def create_image_loader(self, images: torch.Tensor):
        return torch.utils.data.DataLoader(images)
