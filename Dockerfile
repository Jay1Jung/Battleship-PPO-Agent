# syntax=docker/dockerfile:1.7
FROM python:3.10-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.1+cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/out

ENTRYPOINT ["python3", "train_PPO_with_agent.py"]
