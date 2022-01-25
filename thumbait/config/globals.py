import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

MODEL_PATH = os.environ.get("MODEL_PATH", "models/models/ThumbaitLight18View")
MODEL_PATH_TRENDS = os.environ.get(
    "MODEL_PATH_TRENDS", "models/models/ThumbaitLight18Trend"
)
MODEL_TYPE = os.environ.get("MODEL_TYPE", "thumbaitLight")
MODEL_TYPE_TRENDS = os.environ.get("MODEL_TYPE_TRENDS", "thumbaitLight")
API_KEY = os.environ.get("API_KEY", "thumbait18")
