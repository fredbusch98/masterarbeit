#!/bin/bash

# Update package lists
apt-get update && \

# Install required packages
apt-get install -y build-essential nano libjpeg-dev libpng-dev && \

# Upgrade torchvision
pip install --upgrade --no-cache-dir torchvision && \

# Install Python dependencies
pip install pandas datasets scikit-learn trl transformers sacrebleu unsloth && \

# Set compiler environment variable
export CC=/usr/bin/gcc && \

# Start training
python train.py
