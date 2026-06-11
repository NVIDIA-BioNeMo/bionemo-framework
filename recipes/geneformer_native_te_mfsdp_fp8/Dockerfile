# syntax=docker/dockerfile:1.4
FROM nvcr.io/nvidia/pytorch:26.04-py3

RUN apt-get update && apt-get install -y git

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/requirements.txt \
    pip install -r /requirements.txt

WORKDIR /workspace/bionemo
COPY . .
