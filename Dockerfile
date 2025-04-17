# Start with NVIDIA CUDA base image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

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
    gnupg \
    lsb-release \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add ROS 2 repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 packages
RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    numpy

# Set LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH

# Set working directory
WORKDIR /app

# Source ROS environment in bash
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Command to run when container starts
CMD ["/bin/bash"]