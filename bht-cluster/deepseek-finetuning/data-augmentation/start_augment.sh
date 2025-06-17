#!/bin/bash

pip install --upgrade \
  "numpy<2" \
  torch==2.1.2 \
  torchvision==0.16.2 \
  transformers==4.40.0 \
  sentencepiece \
  pandas \
  sacremoses

# Start data augmentation
python back_translation.py
