# Use the official NVIDIA CUDA base image which comes with the CUDA toolkit
FROM nvidia/cuda:12.3.1-base-ubuntu20.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda (this will install Miniconda in /opt/miniconda)
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda && \
    rm /tmp/miniconda.sh

# Add conda to PATH so that conda commands can be used directly
ENV PATH="/opt/miniconda/bin:$PATH"

# Optionally update conda
RUN conda update -n base -c defaults conda -y

# Set the default command to bash
CMD ["bash"]
