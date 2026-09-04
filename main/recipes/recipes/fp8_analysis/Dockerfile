# syntax=docker/dockerfile:1.4
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libfreetype6 libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/bionemo

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .
