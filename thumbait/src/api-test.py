from googleapiclient import discovery
import json

# API information
api_service_name = "youtube"
api_version = "v3"
# API key
DEVELOPER_KEY = "AIzaSyB3ysNIpTKr7U8QWZmcEe9WXlQNOFyUwqM"
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
