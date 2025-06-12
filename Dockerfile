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
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
Copy torch-requirements.txt .
Copy requirements.txt .
# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r torch-requirements.txt.txt && \
    pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY  . .

# Default command to run the bot
CMD ["python", "app.py"]