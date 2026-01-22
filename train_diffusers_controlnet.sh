#!/bin/bash
# SDXL ControlNet Training using diffusers (official HuggingFace example)
# Simpler than kohya, well documented

set -e

TARGET=${1:-normal}

echo "======================================"
echo "SDXL ControlNet Training (diffusers)"
echo "Target: $TARGET"
echo "======================================"

# Install dependencies
pip install accelerate transformers diffusers peft datasets --quiet

# Clone diffusers examples if needed
if [ ! -d "diffusers" ]; then
    echo "Cloning diffusers..."
    git clone https://github.com/huggingface/diffusers.git
fi

# Prepare dataset
echo "Preparing dataset..."
python3 prepare_kohya_dataset.py --target $TARGET

# Count samples
SAMPLE_COUNT=$(ls data/kohya/$TARGET/image | wc -l)
echo "Found $SAMPLE_COUNT samples"

# Train
echo ""
echo "Starting training..."

accelerate launch diffusers/examples/controlnet/train_controlnet_sdxl.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --output_dir="./output/controlnet-sdxl-$TARGET" \
    --train_data_dir="./data/kohya/$TARGET" \
    --conditioning_image_column="conditioning" \
    --image_column="image" \
    --caption_column="text" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --num_train_epochs=20 \
    --learning_rate=1e-5 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=500 \
    --mixed_precision="fp16" \
    --checkpointing_steps=1000 \
    --validation_steps=500 \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --tracker_project_name="pbr-controlnet-sdxl"

echo ""
echo "Training complete!"
echo "Model saved to: ./output/controlnet-sdxl-$TARGET"
