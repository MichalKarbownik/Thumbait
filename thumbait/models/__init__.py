from enum import Enum
import json


from models.thumbait_networks import (
    Thumbait18,
    Thumbait50,
    ThumbLight,
)


class ModelType(str, Enum):
    thumbait18 = "thumbait18"
    thumbait50 = "thumbait50"
    ThumbLight = "ThumbLight"


models = {
    "thumbait18": Thumbait18,
    "thumbait50": Thumbait50,
    "thumbaitLight": ThumbLight,
}


configs = {"thumbaitLight": {}}

with open("models/config/ThumbaitLight18View.json", "r") as file:
    configs["thumbaitLight"]["views"] = json.load(file)

with open("models/config/ThumbaitLight18Trend.json", "r") as file:
    configs["thumbaitLight"]["trends"] = json.load(file)
