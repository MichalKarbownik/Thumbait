from googleapiclient import discovery
import json

from Thumbait.env import YOUTUBE_API_KEY

api_service_name = "youtube"
api_version = "v3"
# API key
DEVELOPER_KEY = YOUTUBE_API_KEY
# API client
youtube = discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)
# 'request' variable is the only thing you must change
# depending on the resource and method you need to use
# in your query
request = youtube.videos().list(part="id,snippet", chart="mostPopular", maxResults=50)
# Query execution
response = request.execute()
# Print the results
with open("dump.json", "w") as file:
    json.dump(response["items"], file)
