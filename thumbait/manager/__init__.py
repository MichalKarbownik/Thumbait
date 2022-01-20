import re

import torch

from models import models, ModelType
from preprocessor import TextPreprocessor, ImagePreprocessor
from connector import ConnectorYouTube
from config.globals import API_KEY
from logger import get_logger


logger = get_logger(__name__)


class ThumbaitManager(TextPreprocessor, ImagePreprocessor, ConnectorYouTube):
    def __init__(
        self,
        model_url: str,
        model_type: str,
        fasttext_path="models/cc.en.300.bin",
        view_max: int = 548866548,
        view_min: int = 0,
    ):
        TextPreprocessor.__init__(self, fasttext_path)
        ImagePreprocessor.__init__(self)
        ConnectorYouTube.__init__(self, API_KEY)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_model = self._load_model(model_url, model_type)
        self.view_max = view_max
        self.view_min = view_min

    def _load_model(
        self,
        model_url: str,
        model_type: ModelType,
    ):
        model = models[model_type](None, self.device).to(self.device)
        checkpoint = torch.load(model_url)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def predict(self, v: str) -> float:
        """
        Predicts video number of Views

        v : str
            youtube video id or youtube link
        """
        # get youtube video id

        if re.search(r"http", v):
            res = re.split(r"=", v)
            v = re.split("&", res[1])[0]

        # image pipeline
        image = self.fetch_image(v)
        image = self.transform_image(image)

        # set loader in model
        self.torch_model.image_data_loader = self.create_image_loader([image])

        # text pipeline
        title, view_count = self.get_statistics(v)

        title_embedd = self.preprocess_texts([title])

        view_count_pred = self.torch_model(title_embedd.to(self.device), [0])

        return {
            "link": f"https://www.youtube.com/watch?v={v}",
            "view_count": int(view_count),
            "view_count_pred": (
                view_count_pred.tolist()[0][0] * (self.view_max - self.view_min)
            )
            + self.view_min,
            "raw_output": view_count_pred.tolist()[0][0],
        }
