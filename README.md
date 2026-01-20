# PBR ControlNet Training

Train ControlNet models to convert basecolor textures to PBR maps (roughness, metallic).

## Overview

This trains two ControlNet models:
- **Roughness ControlNet**: basecolor → roughness map
- **Metallic ControlNet**: basecolor → metallic map

Uses [MatSynth](https://huggingface.co/datasets/gvecchio/MatSynth) dataset (~4000 PBR materials).

## Requirements

- GPU: RTX 3090/4090 (24GB) or A100
- Disk: ~50GB for dataset
- Time: ~2-4 hours per model

## Quick Start (Vast.ai)

### 1. Rent GPU

- **Image**: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel`
- **GPU**: RTX 4090 or A100
- **Disk**: 100GB+

### 2. Clone & Setup

```bash
cd /workspace
git clone https://github.com/P5ina/pbr-controlnet-training.git
cd pbr-controlnet-training
pip install -r requirements.txt
```

### 3. Login (optional)

```bash
wandb login  # For monitoring
```

### 4. Prepare Dataset

```bash
# Both roughness and metallic (~4000 samples each)
python prepare_dataset.py --output ./data --all

# Or test with fewer samples
python prepare_dataset.py --output ./data --all --max-samples 100
```

Dataset structure:
```
data/
├── roughness/
│   ├── conditioning/   (basecolor images)
│   ├── target/         (roughness images)
│   └── prompts.json
└── metallic/
    ├── conditioning/
    ├── target/
    └── prompts.json
```

### 5. Train

```bash
# Train roughness ControlNet
accelerate launch train_controlnet.py --config config.yaml

# Train metallic ControlNet
accelerate launch train_controlnet.py --config config_metallic.yaml

# Or train both
bash train.sh
```

### 6. Monitor

Open [wandb.ai](https://wandb.ai) to see:
- Loss curves
- Validation images (input basecolor → output roughness/metallic)

## Configuration

Edit `config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_map` | roughness | Target map type |
| `batch_size` | 4 | Training batch size |
| `max_train_steps` | 10000 | Total steps |
| `learning_rate` | 1e-5 | Learning rate |
| `validation_steps` | 500 | Validate every N steps |

### Memory Optimization

**RTX 4090 (24GB):**
```yaml
batch_size: 4
gradient_accumulation_steps: 4
gradient_checkpointing: true
```

**RTX 3090 (24GB):**
```yaml
batch_size: 2
gradient_accumulation_steps: 8
```

**16GB VRAM:**
```yaml
batch_size: 1
gradient_accumulation_steps: 16
resolution: 512
```

## Output

```
output/
├── controlnet-roughness-final/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── controlnet-metallic-final/
│   └── ...
└── validation/
    └── step_500/
        ├── 00001_input.png
        └── 00001_output.png
```

## Usage After Training

### ComfyUI

1. Copy model to ComfyUI:
```bash
cp -r output/controlnet-roughness-final ComfyUI/models/controlnet/
cp -r output/controlnet-metallic-final ComfyUI/models/controlnet/
```

2. Use with ControlNet nodes:
```
Load ControlNet Model → Apply ControlNet → KSampler
```

### Python

```python
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
import torch

controlnet = ControlNetModel.from_pretrained(
    "./output/controlnet-roughness-final",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# Generate roughness from basecolor
roughness = pipe(
    prompt="roughness map, pbr texture",
    image=basecolor_image,
    num_inference_steps=20,
).images[0]
```

## Workflow

Full PBR pipeline with trained ControlNets:

```
┌────────────┐
│   FLUX     │ (generates basecolor)
└─────┬──────┘
      │
      ├──────────────────┬──────────────────┐
      ▼                  ▼                  ▼
┌───────────┐     ┌────────────┐     ┌────────────┐
│ DeepBump  │     │ ControlNet │     │ ControlNet │
│           │     │ (roughness)│     │ (metallic) │
└─────┬─────┘     └─────┬──────┘     └─────┬──────┘
      │                 │                  │
      ▼                 ▼                  ▼
   Normal           Roughness          Metallic
   + Height
```

## License

MIT License

## Credits

- [MatSynth Dataset](https://huggingface.co/datasets/gvecchio/MatSynth) (CC-BY)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [diffusers](https://github.com/huggingface/diffusers)
