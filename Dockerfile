# Dockerfile
FROM python:3.10.19-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install dvc   # Install DVC

# Copy project code
COPY . /app

# Expose FastAPI port
EXPOSE 8000

# Pull DVC data/models at container runtime, then start app
CMD dvc remote modify dagshub_remote --local user $DAGSHUB_USER && \
    dvc remote modify dagshub_remote --local password $DAGSHUB_TOKEN && \
    dvc pull && \
    uvicorn main:app --host 0.0.0.0 --port 8000
