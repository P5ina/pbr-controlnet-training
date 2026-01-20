#!/bin/bash
# Train ControlNet for PBR map generation

set -e

echo "=== PBR ControlNet Training ==="

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Login to services
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY"
fi

# Prepare datasets (both roughness and metallic)
echo "Preparing datasets..."
python prepare_dataset.py --output ./data --all --max-samples 4000

# Train roughness ControlNet
echo "Training roughness ControlNet..."
accelerate launch train_controlnet.py --config config.yaml

# Train metallic ControlNet
echo "Training metallic ControlNet..."
accelerate launch train_controlnet.py --config config_metallic.yaml

echo "Training complete!"
echo "Models saved to ./output/"
