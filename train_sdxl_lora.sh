#!/bin/bash
# SDXL LoRA Training for PBR Maps

set -e

echo "======================================"
echo "SDXL LoRA PBR Training Pipeline"
echo "======================================"

# Install dependencies
pip install peft accelerate transformers diffusers torch torchvision --quiet

# Check if dataset exists
if [ ! -d "data/chained/basecolor" ]; then
    echo "Error: Dataset not found at data/chained/"
    echo "Run prepare_dataset_chained.py first"
    exit 1
fi

SAMPLE_COUNT=$(ls data/chained/basecolor | wc -l)
echo "Found $SAMPLE_COUNT samples"

# Train all three maps
echo ""
echo "[1/3] Training Normal LoRA..."
python3 train_sdxl_lora.py --config config_sdxl_lora.yaml --target normal

echo ""
echo "[2/3] Training Roughness LoRA..."
python3 train_sdxl_lora.py --config config_sdxl_lora.yaml --target roughness

echo ""
echo "[3/3] Training Metallic LoRA..."
python3 train_sdxl_lora.py --config config_sdxl_lora.yaml --target metallic

echo ""
echo "======================================"
echo "Training complete!"
echo "======================================"
echo "LoRA weights saved to:"
echo "  output/lora-normal-final/"
echo "  output/lora-roughness-final/"
echo "  output/lora-metallic-final/"
