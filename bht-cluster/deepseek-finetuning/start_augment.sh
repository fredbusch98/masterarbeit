#!/bin/bash

# Install required packages
pip install --upgrade torch transformers sentencepiece pandas sacremoses

# Start data augmentation
python back_translation.py
