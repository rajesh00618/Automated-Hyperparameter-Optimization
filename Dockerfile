FROM python:3.9-slim

# Install system dependencies for matplotlib and other packages
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY notebooks/ ./notebooks/

RUN mkdir -p /app/outputs

ENV MLFLOW_TRACKING_URI=file:///app/outputs/mlruns

CMD ["python", "src/optimize.py"]
