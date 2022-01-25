FROM python:3.9

WORKDIR /usr/src/app

# Install necessary data
COPY ./requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader omw-1.4


# start fastapi server. Might be helfull
CMD ["python", "server.py"]
