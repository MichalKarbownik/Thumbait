from enum import Enum
from models.thumbait_networks import Thumbait18, Thumbait50


class ModelType(str, Enum):
    thumbait18 = "thumbait18"
    thumbait50 = "thumbait50"


models = {"thumbait18": Thumbait18, "thumbait50": Thumbait50}
