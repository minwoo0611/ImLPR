# Start from the specified PyTorch base image
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

# Update and install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    libopenblas-dev \
    libopenexr-dev \
    libeigen3-dev \
    libgl1-mesa-glx \
    libxext6 \
    libxrender-dev \
    python3-setuptools \
    python3-pip \
    mpich \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools==59.5.0
RUN conda install -c conda-forge mpi4py mpich -y
# Install Python packages
RUN pip install \
    pandas \
    tensorboard \
    numpy==1.24.1 \
    laspy \
    pytest \
    addict \
    pytorch-metric-learning==0.9.97 \
    yapf==0.40.1 \
    h5py \
    transforms3d \
    open3d \
    tqdm \
    einops\
    timm \
    termcolor
    

RUN pip install bagpy \
    utm \
    pyexr \
    pyntcloud \
    Jinja2==3.1.5 \
    ninja==1.11.1.1 \
    scikit-learn \
    torchtyping \
    future_fstrings \
    pytorch_metric_learning==1.1.2 \
    psutil \
    tensorboardX \
    torchpack==0.3.1

RUN pip3 install -U xformers==0.0.29.post2 --index-url https://download.pytorch.org/whl/cu118

# Set the default command
CMD ["bash"]
