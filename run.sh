#!/bin/bash

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train
python3 train.py --data-dir dataset --output-dir artifacts --epochs 12 --batch-size 32

# Predict
python3 predict.py --model artifacts/best_model.pt --image dataset/000000/000000_000000_000000.jpg --topk 3