# syntax=docker/dockerfile:1.4
FROM nvcr.io/nvidia/pytorch:26.04-py3

RUN apt-get update && apt-get install -y tmux npm

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/requirements.txt \
    PIP_CONSTRAINT= pip install -r /requirements.txt

RUN curl -fsSL https://claude.ai/install.sh | bash  # Install Claude CLI tool
RUN npm install -g @openai/codex

WORKDIR /workspace/bionemo
COPY . .
