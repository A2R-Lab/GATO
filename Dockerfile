# start with NVIDIA CUDA base image and ROS Humble
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS cuda
FROM ros:humble-ros-base

# CUDA
COPY --from=cuda /usr/local/cuda /usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10

# install system dependencies
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

# set python aliases
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    numpy

# install MuJoCo
RUN git clone https://github.com/deepmind/mujoco.git \
    && cd mujoco \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=/usr .. \
    && cmake --build . \
    && cmake --install . \
    && cd ../.. \
    && rm -rf mujoco

# set LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH

# set working directory
WORKDIR /workspace

# source ROS environment in bash
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "[ -f /workspace/install/setup.bash ] && source /workspace/install/setup.bash" >> ~/.bashrc
RUN echo "[ -f /workspace/.venv/bin/activate ] && source /workspace/.venv/bin/activate" >> ~/.bashrc

# command to run when container starts
CMD ["/bin/bash"]