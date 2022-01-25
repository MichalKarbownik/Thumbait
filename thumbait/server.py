import os
from typing import Union

from fastapi.responses import JSONResponse  # type: ignore
from fastapi.encoders import jsonable_encoder  # type: ignore
import uvicorn  # type: ignore

from config.app import app
from logger import get_logger
from config import models
from config.globals import MODEL_PATH, MODEL_PATH_TRENDS, MODEL_TYPE, MODEL_TYPE_TRENDS
from manager import ThumbaitManager

logger = get_logger(__name__)


manager = ThumbaitManager(
    model_path=MODEL_PATH,
    model_arch=MODEL_TYPE,
    _type="light",
    model_path_trends=MODEL_PATH_TRENDS,
    model_arch_trends=MODEL_TYPE_TRENDS,
)


@app.get("/", response_model=models.Status)
def check_server() -> JSONResponse:
    """Checks if server is functioning properly"""
    logger.info("Works")
    return JSONResponse({"success": True}, 200)


@app.get("/predict", response_model=models.Songs)
def get_data_song_raw(v: str) -> JSONResponse:
    """
    Predicts video number of Views

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
    response = manager.predict(v)
    return JSONResponse(jsonable_encoder(response), 200)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=False)
