#!/bin/bash
# =============================================================================
# Cloud GPU Setup Script for Multi-task PBR Training
# Compatible with: RunPod, Vast.ai, Lambda Labs, Paperspace
# =============================================================================

set -e

echo "========================================"
echo "Multi-task PBR Training - Cloud Setup"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root (common on cloud instances)
if [ "$EUID" -eq 0 ]; then
    PIP="pip"
else
    PIP="pip"
fi

# =============================================================================
# 1. System packages
# =============================================================================
echo -e "\n${YELLOW}[1/6] Installing system packages...${NC}"

if command -v apt-get &> /dev/null; then
    apt-get update -qq
    apt-get install -y -qq git wget unzip htop tmux nvtop 2>/dev/null || true
elif command -v yum &> /dev/null; then
    yum install -y -q git wget unzip htop tmux 2>/dev/null || true
fi

echo -e "${GREEN}✓ System packages installed${NC}"

# =============================================================================
# 2. Python dependencies
# =============================================================================
echo -e "\n${YELLOW}[2/6] Installing Python dependencies...${NC}"

# Skip pip upgrade (causes issues on some systems)
# $PIP install --upgrade pip -q

$PIP install -q \
    torch \
    torchvision \
    tqdm \
    pyyaml \
    wandb \
    pillow \
    numpy \
    requests \
    datasets \
    || $PIP install \
    torch \
    torchvision \
    tqdm \
    pyyaml \
    wandb \
    pillow \
    numpy \
    requests \
    datasets

echo -e "${GREEN}✓ Python dependencies installed${NC}"

# =============================================================================
# 3. Verify GPU
# =============================================================================
echo -e "\n${YELLOW}[3/6] Checking GPU...${NC}"

python3 << 'EOF'
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✓ GPU: {gpu_name}")
    print(f"✓ VRAM: {gpu_mem:.1f} GB")

    # Recommend batch size
    if gpu_mem >= 40:
        print(f"✓ Recommended batch size: 16")
    elif gpu_mem >= 20:
        print(f"✓ Recommended batch size: 8")
    elif gpu_mem >= 12:
        print(f"✓ Recommended batch size: 4")
    else:
        print(f"✓ Recommended batch size: 2")
else:
    print("✗ No GPU detected!")
    exit(1)
EOF

echo -e "${GREEN}✓ GPU verified${NC}"

# =============================================================================
# 4. Setup directory structure
# =============================================================================
echo -e "\n${YELLOW}[4/6] Setting up directories...${NC}"

mkdir -p data/materials/{basecolor,normal,roughness,metallic,height}
mkdir -p output/multitask
mkdir -p models

echo -e "${GREEN}✓ Directories created${NC}"

# =============================================================================
# 5. Download sample data (optional)
# =============================================================================
echo -e "\n${YELLOW}[5/6] Data setup...${NC}"

if [ -d "data/materials/basecolor" ] && [ "$(ls -A data/materials/basecolor 2>/dev/null)" ]; then
    echo "✓ Data already exists in data/materials/"
    MATERIAL_COUNT=$(ls data/materials/basecolor | wc -l)
    echo "  Found $MATERIAL_COUNT materials"
else
    echo ""
    echo "No data found. You have 4 options:"
    echo ""
    echo "  1. Upload your own data to: data/materials/"
    echo "     Expected structure:"
    echo "       data/materials/basecolor/*.jpg"
    echo "       data/materials/normal/*.jpg"
    echo "       data/materials/roughness/*.jpg"
    echo "       data/materials/metallic/*.jpg"
    echo "       data/materials/height/*.jpg (optional)"
    echo ""
    echo "  2. Download from ambientCG (CC0 license):"
    echo "     ./download_data.sh 500"
    echo ""
    echo "  3. Download from MatSynth (~4000 high-quality PBR materials):"
    echo "     ./download_matsynth.sh 1000"
    echo ""
    echo "  4. Convert existing PBR dataset:"
    echo "     python prepare_multitask_data.py convert --input /path/to/materials --output data/materials"
    echo ""
fi

# =============================================================================
# 6. Create helper scripts
# =============================================================================
echo -e "\n${YELLOW}[6/6] Creating helper scripts...${NC}"

# Training script
cat > train.sh << 'TRAIN_EOF'
#!/bin/bash
# Start training with optional wandb login

if [ "$1" == "--wandb" ]; then
    echo "Logging in to Weights & Biases..."
    wandb login
fi

echo "Starting training..."
echo "Monitor with: tail -f output/multitask/training.log"
echo "Or use: watch -n 5 nvidia-smi"
echo ""

# Run in tmux for persistence
if command -v tmux &> /dev/null; then
    tmux new-session -d -s training "python train_multitask.py --config config_multitask.yaml 2>&1 | tee output/multitask/training.log"
    echo "Training started in tmux session 'training'"
    echo "Attach with: tmux attach -t training"
    echo "Detach with: Ctrl+B, then D"
else
    python train_multitask.py --config config_multitask.yaml 2>&1 | tee output/multitask/training.log
fi
TRAIN_EOF
chmod +x train.sh

# Validation script
cat > validate.sh << 'VALIDATE_EOF'
#!/bin/bash
python prepare_multitask_data.py validate --data data/materials
VALIDATE_EOF
chmod +x validate.sh

# Download data script (ambientCG)
cat > download_data.sh << 'DOWNLOAD_EOF'
#!/bin/bash
MAX_MATERIALS=${1:-500}
echo "Downloading up to $MAX_MATERIALS materials from ambientCG..."
python prepare_multitask_data.py download --output data/materials --max $MAX_MATERIALS --resolution 512
python prepare_multitask_data.py validate --data data/materials
DOWNLOAD_EOF
chmod +x download_data.sh

# Download MatSynth data script
cat > download_matsynth.sh << 'MATSYNTH_EOF'
#!/bin/bash
MAX_MATERIALS=${1:-1000}
CC0_FLAG=""
if [ "$2" == "--cc0" ]; then
    CC0_FLAG="--cc0-only"
    echo "Downloading up to $MAX_MATERIALS CC0-only materials from MatSynth..."
else
    echo "Downloading up to $MAX_MATERIALS materials from MatSynth..."
fi

# Install datasets library if not present
pip install -q datasets

python download_matsynth.py --output data/materials --max $MAX_MATERIALS --resolution 512 $CC0_FLAG
python prepare_multitask_data.py validate --data data/materials
MATSYNTH_EOF
chmod +x download_matsynth.sh

# Monitor script
cat > monitor.sh << 'MONITOR_EOF'
#!/bin/bash
echo "=== GPU Status ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
echo ""
echo "=== Training Log (last 20 lines) ==="
tail -20 output/multitask/training.log 2>/dev/null || echo "No training log yet"
MONITOR_EOF
chmod +x monitor.sh

# Export model script
cat > export_model.sh << 'EXPORT_EOF'
#!/bin/bash
echo "Copying best model to models/..."
if [ -f "output/multitask/best/model.pth" ]; then
    cp output/multitask/best/model.pth models/pbr_multitask.pth
    echo "✓ Model exported to models/pbr_multitask.pth"
    ls -lh models/pbr_multitask.pth
else
    echo "✗ No best model found. Train first!"
fi
EXPORT_EOF
chmod +x export_model.sh

echo -e "${GREEN}✓ Helper scripts created${NC}"

# =============================================================================
# Done!
# =============================================================================
echo ""
echo "========================================"
echo -e "${GREEN}Setup complete!${NC}"
echo "========================================"
echo ""
echo "Quick start:"
echo ""
echo "  1. Add your data:     Upload to data/materials/ or run ./download_data.sh"
echo "  2. Validate data:     ./validate.sh"
echo "  3. Start training:    ./train.sh"
echo "  4. Monitor progress:  ./monitor.sh"
echo "  5. Export model:      ./export_model.sh"
echo ""
echo "Config file: config_multitask.yaml"
echo "  - Adjust batch_size based on your GPU VRAM"
echo "  - Set use_wandb: false if you don't want logging"
echo ""
echo "Useful commands:"
echo "  tmux attach -t training    # Attach to training session"
echo "  nvidia-smi                 # Check GPU usage"
echo "  htop                       # Check CPU/RAM usage"
echo ""
