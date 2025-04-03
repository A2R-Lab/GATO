# Start with NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    torchaudio

# Set working directory
WORKDIR /app

# Command to run when container starts
CMD ["/bin/bash"]