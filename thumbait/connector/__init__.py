import shutil
from PIL import Image
from PIL.ImageFile import ImageFile

from googleapiclient import discovery
from googleapiclient.errors import HttpError
import requests

from logger import get_logger
from utils.text_utils import _ensure_exists

logger = get_logger(__name__)


class ConnectorYouTube:
    def __init__(
        self,
        API_KEY: str,
        api_service_name: str = "youtube",
        api_version: str = "v3",
        columns=[
            "video_id",
            "title",
            "publishedAt",
            "channelId",
            "channelTitle",
            "view_count",
            "likes",
            "comment_count",
            "thumbnail_link",
            "description",
        ],
    ):
        self.youtube = discovery.build(
            api_service_name, api_version, developerKey=API_KEY
        )
        self.columns = columns

    def fetch_image(self, v: str) -> ImageFile:
        _ensure_exists("tmp/image")

        path = f"tmp/image/{v}.jpg"
        url = f"https://img.youtube.com/vi/{v}/hqdefault.jpg"
        response = requests.get(url, stream=True)
        with open(path, "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)
        image = Image.open(path)
        return image

    def get_statistics(self, v: str) -> tuple[str, int]:
        video_data = (
            self.youtube.videos()
            .list(part="snippet,statistics", id=v)
            .execute()["items"][0]
        )
        title = video_data["snippet"]["title"]
        view_count = video_data["statistics"]["viewCount"]

        return title, view_count
