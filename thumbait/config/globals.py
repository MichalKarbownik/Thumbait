import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

MODEL_URL = os.environ.get("MODEL_URL", "models/BaseModel18")
MODEL_TYPE = os.environ.get("MODEL_TYPE", "thumbait18")
API_KEY = os.environ.get("API_KEY", "thumbait18")
