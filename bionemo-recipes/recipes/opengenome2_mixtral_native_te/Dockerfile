# syntax=docker/dockerfile:1.4
FROM nvcr.io/nvidia/pytorch:26.03-py3

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/requirements.txt \
    PIP_CONSTRAINT= pip install -r /requirements.txt

WORKDIR /workspace/bionemo
COPY . .
