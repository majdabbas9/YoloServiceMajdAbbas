# Base image
FROM python:3.11-slim AS base

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip && pip install -r /app/torch-requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt
EXPOSE 8080
# Default command to run the bot
CMD ["python3","app"]
