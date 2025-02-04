# Use an official Ubuntu image as the base
FROM ubuntu:20.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda && \
    rm /tmp/miniconda.sh

# Add conda to PATH
ENV PATH="/opt/miniconda/bin:$PATH"

# (Optional) Update conda
RUN conda update -n base -c defaults conda

# Set the working directory (optional)
WORKDIR /workspace

# Set the default command to bash
CMD ["bash"]
