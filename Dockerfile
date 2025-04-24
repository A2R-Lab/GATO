# start with NVIDIA CUDA base image and ROS Humble
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS cuda
FROM ros:humble-ros-base

# CUDA
COPY --from=cuda /usr/local/cuda /usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

# environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10

# system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-numpy \
    python3-pip \
    vim \
    gnupg \
    lsb-release \
    software-properties-common \
    ros-humble-urdfdom \
    ros-humble-hpp-fcl \
    ros-humble-urdfdom-headers \
    python3-colcon-common-extensions \
    python3-rosdep \
    libxinerama-dev \
    libglfw3-dev \
    libxcursor-dev \
    libxi-dev \
    libxrandr-dev \
    libxxf86vm-dev \
    x11-apps \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libxfixes-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# python aliases
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
&& ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# PyTorch
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    numpy

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH}

# install MuJoCo
RUN git clone https://github.com/deepmind/mujoco.git \
    && cd mujoco \
    && mkdir build \
    && cd build \
    && cmake .. \
    && cmake --build . \
    && cmake --install . 

# set working directory
WORKDIR /workspace

# auto source ROS2
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
#RUN echo "[ -f /workspace/install/setup.bash ] && source /workspace/install/setup.bash" >> ~/.bashrc

# auto source python environment
RUN echo "[ -f /workspace/.venv/bin/activate ] && source /workspace/.venv/bin/activate" >> ~/.bashrc

# when container starts
CMD ["/bin/bash"]