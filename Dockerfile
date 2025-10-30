FROM python:3.11-slim-buster

RUN apt-get update && apt-get install -y libgomp1

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]