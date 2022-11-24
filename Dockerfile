FROM python:3.9-slim-buster

RUN apt-get update \
    && apt-get install -y git \
    && apt-get clean \
    && apt-get install libpq-dev wget gnupg -y

RUN wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - \
    && apt-get update \
    && apt-get -y install postgresql

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ./app project/app
COPY ./main.py project/main.py
COPY ./client.py project/client.py

WORKDIR /project