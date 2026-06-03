FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 python3.8-dev python3-pip \
    build-essential cmake git \
    libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python3.8 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Upgrade pip
RUN pip install --upgrade pip setuptools==57.5.0 wheel

# Install core libraries
RUN pip install numpy matplotlib pillow

# Install PyTorch (CUDA)
RUN pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install gym (compatible version)
RUN pip install gym==0.23.1 torch-geometric

# Install CityFlow
RUN git clone https://github.com/cityflow-project/CityFlow.git
RUN cd CityFlow && pip install .



# Set working directory
WORKDIR /app

# Default command
CMD ["/bin/bash"]