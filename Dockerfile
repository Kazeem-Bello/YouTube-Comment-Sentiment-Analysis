FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y libgomp1

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

RUN python3 -m nltk.downloader stopwords wordnet omw-1.4

CMD ["python3", "app.py"]