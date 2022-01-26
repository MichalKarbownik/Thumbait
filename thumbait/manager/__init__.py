import re
from typing import Optional

import torch

from models import models, configs, ModelType
from preprocessor import TextPreprocessor, ImagePreprocessor
from connector import ConnectorYouTube
from config.globals import API_KEY
from logger import get_logger


logger = get_logger(__name__)


class ThumbaitManager(TextPreprocessor, ImagePreprocessor, ConnectorYouTube):
    def __init__(
        self,
        model_path: str,
        model_arch: str,
        fasttext_path="models/cc.en.300.bin",
        view_max: int = 548866548,
        view_min: int = 0,
        view_mean: float = 2442596.3483247557,
        view_std: float = 8797020.75652489,
        _type: str = "big",
        model_path_trends: Optional[str] = None,
        model_arch_trends: Optional[str] = None,
    ):
        TextPreprocessor.__init__(self, _type, fasttext_path)
        ImagePreprocessor.__init__(self)
        ConnectorYouTube.__init__(self, API_KEY)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.torch_model = self._load_model(model_path, model_arch, "views")

        if self._type == "light":
            self.torch_model_trends = self._load_model(
                model_path_trends, model_arch_trends, "trends"
            )

        self.view_max = view_max
        self.view_min = view_min
        self.view_mean = view_mean
        self.view_std = view_std

    def _load_model(self, model_path: str, model_arch: ModelType, _pred: str = "views"):
        kwargs = None
        if model_arch in configs:
            kwargs = configs[model_arch][_pred]

        model = models[model_arch](kwargs, self.device).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def predict(self, v: str) -> float:
        """
        Predicts video number of Views / Trending score

        Parameters
        --------------------------------
        v : str
            youtube video id or youtube link

        Returns
        --------------------------------

        dict: {
            "link": str,
            "view_count": int,
            "view_count_pred": float,
            "raw_output_view": float
            "output_trend": float
        }

        """
        # get youtube video id

        if re.search(r"youtu.be", v):
            res = re.split(r".be/", v)
            v = res[1]

        elif re.search(r"http", v):
            res = re.split(r"=", v)
            v = re.split("&", res[1])[0]

        # image pipeline
        image = self.fetch_image(v)
        image = self.transform_image(image)

        image_data_loader = self.create_image_loader([image])  # set loader in model

        # text pipeline
        title, view_count = self.get_statistics(v)

        title_embedd = self.preprocess_texts([title])
        trending_pred = torch.zeros(1, 1)
        if self._type == "big":
            self.torch_model.image_data_loader = image_data_loader
            with torch.no_grad():
                view_count_pred = self.torch_model(title_embedd.to(self.device), [0])
            out_views = (
                view_count_pred.tolist()[0][0] * (self.view_max - self.view_min)
                + self.view_min
            )

        if self._type == "light":
            with torch.no_grad():
                view_count_pred = self.torch_model(
                    title_embedd.to(self.device),
                    next(
                        iter(image_data_loader),
                    ).to(self.device),
                )
                trending_pred = self.torch_model_trends(
                    title_embedd.to(self.device),
                    next(iter(image_data_loader)).to(self.device),
                )

            out_views = view_count_pred.tolist()[0][0] * self.view_std + self.view_mean

        return {
            "link": f"https://www.youtube.com/watch?v={v}",
            "view_count": int(view_count),
            "view_count_pred": out_views,
            "raw_output": view_count_pred.tolist()[0][0],
            "output_trend": torch.argmax(trending_pred, dim=1).tolist()[0],
        }
