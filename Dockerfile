FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.1+cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "train_PPO_with_agent.py"]
