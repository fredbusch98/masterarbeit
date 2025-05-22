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
pip install pandas datasets scikit-learn trl transformers sacrebleu rouge-score unsloth && \

# Set compiler environment variable
export CC=/usr/bin/gcc && \

# Get train_dir from argument
TRAIN_DIR="$1"

# Get num_epochs from argument
NUM_EPOCHS="$2"

# Start training with the provided train_dir
python train_deepseek_distill.py --train_dir "$TRAIN_DIR" --num_epochs "$NUM_EPOCHS"
