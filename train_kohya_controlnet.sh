#!/bin/bash
# SDXL ControlNet Training using kohya-ss/sd-scripts

set -e

echo "======================================"
echo "SDXL ControlNet Training (kohya-ss)"
echo "======================================"

# Clone kohya-ss if not exists
if [ ! -d "sd-scripts" ]; then
    echo "Cloning kohya-ss/sd-scripts..."
    git clone https://github.com/kohya-ss/sd-scripts.git
    cd sd-scripts
    pip install -r requirements.txt
    cd ..
fi

# Check dataset
if [ ! -d "data/chained/basecolor" ]; then
    echo "Error: Dataset not found. Run prepare_dataset_chained.py first"
    exit 1
fi

SAMPLE_COUNT=$(ls data/chained/basecolor | wc -l)
echo "Found $SAMPLE_COUNT samples"

# Prepare dataset in kohya format
echo "Preparing dataset for kohya..."
python3 prepare_kohya_dataset.py --target normal

# Train ControlNet
echo ""
echo "Starting ControlNet training..."

accelerate launch sd-scripts/sdxl_train_control_net.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --output_dir="./output/controlnet-sdxl-normal" \
    --output_name="controlnet-sdxl-normal" \
    --train_data_dir="./data/kohya/normal" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=1e-5 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=100 \
    --max_train_epochs=20 \
    --mixed_precision="fp16" \
    --save_every_n_epochs=5 \
    --save_model_as="safetensors" \
    --gradient_checkpointing \
    --xformers \
    --cache_latents \
    --conditioning_data_dir="./data/kohya/normal/conditioning"

echo "Training complete!"
