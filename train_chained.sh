#!/bin/bash
# Chained PBR Training Pipeline
# CHORD-style sequential generation: basecolor → normal → roughness → metallic

set -e

echo "======================================"
echo "Chained PBR Training Pipeline"
echo "======================================"

# Install dependencies if needed
if ! pip show diffusers > /dev/null 2>&1; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Login to wandb if needed
if command -v wandb &> /dev/null; then
    echo "Checking wandb login..."
    wandb login --relogin 2>/dev/null || true
fi

# Prepare dataset with ALL maps
echo ""
echo "[Step 1/2] Preparing chained dataset..."
if [ ! -d "data/chained" ]; then
    python prepare_dataset_chained.py \
        --output ./data \
        --max-samples 4000 \
        --resolution 512 \
        --num-workers 8
else
    echo "Dataset already exists at data/chained, skipping..."
fi

# Train all stages sequentially
echo ""
echo "[Step 2/2] Training all stages..."
python train_pix2pix_chained.py --config config_pix2pix_chained.yaml --stage all

echo ""
echo "======================================"
echo "Training complete!"
echo "======================================"
echo ""
echo "Models saved to:"
echo "  output/chained-normal-final/generator.pth"
echo "  output/chained-roughness-final/generator.pth"
echo "  output/chained-metallic-final/generator.pth"
echo ""
echo "To use in ComfyUI, copy models to:"
echo "  ComfyUI/models/pix2pix_chained/"
