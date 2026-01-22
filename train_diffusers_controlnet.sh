#!/bin/bash
# SDXL ControlNet Training using diffusers (official HuggingFace example)
# Simpler than kohya, well documented

set -e

TARGET=${1:-normal}

echo "======================================"
echo "SDXL ControlNet Training (diffusers)"
echo "Target: $TARGET"
echo "======================================"

# Install diffusers from source (required for examples)
pip install accelerate transformers peft datasets wandb --quiet

# Remove old diffusers and install from source
pip uninstall diffusers -y 2>/dev/null || true
pip install git+https://github.com/huggingface/diffusers.git --quiet

# Clone diffusers repo for examples (must match installed version)
if [ -d "diffusers" ]; then
    echo "Removing old diffusers repo..."
    rm -rf diffusers
fi
echo "Cloning diffusers..."
git clone --depth 1 https://github.com/huggingface/diffusers.git

# Install example requirements
pip install -r diffusers/examples/controlnet/requirements_sdxl.txt --quiet 2>/dev/null || true

# Patch the training script to cast conditioning_image column to Image after loading
# This is needed because imagefolder only auto-converts the main image column
echo "Patching diffusers training script..."
TRAIN_SCRIPT="diffusers/examples/controlnet/train_controlnet_sdxl.py"

# Add import for Image feature at the top
sed -i 's/from datasets import load_dataset/from datasets import load_dataset, Image as ImageFeature/' "$TRAIN_SCRIPT"

# Add cast_column after dataset loading - find the line after load_dataset and add casting
# The pattern: after "dataset = load_dataset(" block ends with ")", add the cast
python3 << 'PATCH_EOF'
import re

with open("diffusers/examples/controlnet/train_controlnet_sdxl.py", "r") as f:
    content = f.read()

# Find the get_train_dataset function and add column casting after dataset loading
# We need to add this after the dataset is loaded but before it's used

# Pattern to find where dataset["train"] is first accessed after loading
# Add casting right after the dataset is loaded

patch_code = '''
    # Cast conditioning_image column to Image if it's a string path
    # This is needed for local datasets using imagefolder format
    if "train" in dataset:
        _cond_col = args.conditioning_image_column or "conditioning_image"
        if _cond_col in dataset["train"].column_names:
            _first_item = dataset["train"][0][_cond_col]
            if isinstance(_first_item, str):
                logger.info(f"Casting {_cond_col} column to Image feature")
                dataset["train"] = dataset["train"].cast_column(_cond_col, ImageFeature())
'''

# Find the line "column_names = dataset["train"].column_names" and insert before it
target_line = 'column_names = dataset["train"].column_names'
if target_line in content:
    content = content.replace(target_line, patch_code + "\n    " + target_line)
    print("Dataset patch applied successfully")
else:
    print("Warning: Could not find target line for dataset patch")

# Patch 2: Cast encoder_hidden_states to weight_dtype in UNet call
# The prompt_ids (encoder_hidden_states) are float32 but UNet is fp16
# Find encoder_hidden_states=batch["prompt_ids"] and add .to(weight_dtype)

old_encoder = 'encoder_hidden_states=batch["prompt_ids"],'
new_encoder = 'encoder_hidden_states=batch["prompt_ids"].to(dtype=weight_dtype),'
if old_encoder in content:
    content = content.replace(old_encoder, new_encoder)
    print("encoder_hidden_states dtype patch applied")
else:
    print("Warning: Could not find encoder_hidden_states line")

with open("diffusers/examples/controlnet/train_controlnet_sdxl.py", "w") as f:
    f.write(content)
PATCH_EOF

# Prepare dataset (imagefolder format with metadata.jsonl)
echo "Preparing dataset..."
rm -rf "./data/kohya/$TARGET"
python3 prepare_kohya_dataset.py --target $TARGET

# Count samples
SAMPLE_COUNT=$(ls ./data/kohya/$TARGET/images 2>/dev/null | wc -l)
echo "Found $SAMPLE_COUNT samples"

# Install 8-bit adam
pip install bitsandbytes --quiet

# Train
echo ""
echo "Starting training..."

# Help with memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch "$TRAIN_SCRIPT" \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --output_dir="./output/controlnet-sdxl-$TARGET" \
    --dataset_name="./data/kohya/$TARGET" \
    --conditioning_image_column="conditioning_image" \
    --image_column="image" \
    --caption_column="text" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --num_train_epochs=20 \
    --learning_rate=1e-5 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=500 \
    --mixed_precision="fp16" \
    --checkpointing_steps=1000 \
    --validation_steps=500 \
    --validation_prompt="normal map, stone brick wall texture, blue purple normal map, pbr texture, seamless tileable" \
    --validation_image="./data/kohya/$TARGET/conditioning_images/00000.jpg" \
    --use_8bit_adam \
    --gradient_checkpointing \
    --report_to=wandb \
    --tracker_project_name="pbr-controlnet-sdxl"

echo ""
echo "Training complete!"
echo "Model saved to: ./output/controlnet-sdxl-$TARGET"
