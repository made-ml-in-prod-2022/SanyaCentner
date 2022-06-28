FROM python:3.8-slim

RUN mkdir /build/

COPY requirements.txt /build/requirements.txt

RUN pip install -r /build/requirements.txt

COPY preprocess.py /preprocess.py
