# Use official Python image
FROM python:3.8-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libboost-all-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support (adjust for your CUDA version)
# For CPU only, remove this and use requirements.txt version
RUN pip install --no-cache-dir \
    torch==1.12.1+cu113 \
    torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Copy project files
COPY . .

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/workspace/src"

# Create directories for outputs
RUN mkdir -p /workspace/data/demonstrations \
    /workspace/data/processed \
    /workspace/models/bc \
    /workspace/models/rl \
    /workspace/results/figures \
    /workspace/results/videos \
    /workspace/results/logs

# Default command
CMD ["/bin/bash"]
