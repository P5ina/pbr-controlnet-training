#!/bin/bash
# Train Pix2Pix for PBR map generation

set -e

echo "=== PBR Pix2Pix Training ==="

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Login to wandb
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY"
fi

# Prepare datasets if not already done
if [ ! -d "./data/roughness" ]; then
    echo "Preparing datasets..."
    python prepare_dataset.py --output ./data --all --max-samples 4000
fi

# Train roughness
echo "Training roughness Pix2Pix..."
python train_pix2pix.py --config config_pix2pix.yaml

# Train metallic
echo "Training metallic Pix2Pix..."
python train_pix2pix.py --config config_pix2pix_metallic.yaml

echo "Training complete!"
echo "Models saved to ./output/"
