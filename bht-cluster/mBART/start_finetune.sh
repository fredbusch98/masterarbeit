#!/bin/bash

# Exit on any error
set -e

# Update package lists
apt-get update && \

# Install required packages
apt-get install -y build-essential nano libjpeg-dev libpng-dev && \

# Upgrade torchvision
pip install --upgrade --no-cache-dir torchvision && \

# Install Python dependencies
pip install pandas sentencepiece torch transformers sacrebleu tqdm "accelerate>=0.26.0" && \

# Set compiler environment variable
export CC=/usr/bin/gcc && \

# Start training
python train_mbart.py
