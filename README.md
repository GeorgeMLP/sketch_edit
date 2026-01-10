# Sketch-Guided Image Editing

A modular pipeline for synthetic dataset generation, model training, and evaluation for image editing guided by language and sketch.

## Overview

This codebase provides tools for:
- **Dataset Generation**: Create synthetic datasets for sketch+text-guided image editing
- **Model Training**: Fine-tune diffusion models for multimodal image editing
- **Evaluation**: Evaluate model predictions using multiple metrics (MSE, CLIP similarity, FID)

## Installation

```bash
# Create conda environment
conda create -n sketch_edit python=3.11 -y
conda activate sketch_edit

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install the package in development mode
pip install -e ".[all]"
```

## Quick Start

### 1. Generate Dataset

```bash
# Generate editing instructions using Gemma-3
python scripts/generate_dataset.py \
    --task instructions \
    --data_dir ./dataset \
    --model_id google/gemma-3-4b-it

# Generate inpainted images (remove objects from source images)
python scripts/generate_dataset.py \
    --task inpainting \
    --data_dir ./dataset

# Create dataset manifest (JSON file mapping all data paths)
python scripts/generate_dataset.py \
    --task manifest \
    --data_dir ./dataset \
    --output_file dataset.json
```

### 2. Train a Model

```bash
# Fine-tune InstructPix2Pix
python scripts/train.py \
    --model instruct_pix2pix \
    --dataset_json dataset.json \
    --output_dir ./checkpoints \
    --batch_size 4 \
    --max_steps 1000 \
    --learning_rate 5e-5
```

### 3. Evaluate

```bash
# Evaluate model predictions
python scripts/evaluate.py \
    --predictions_dir ./predictions \
    --ground_truth_dir ./ground_truth \
    --metrics mse clip_sim fid \
    --output_file results.json
```

### 4. Inference

```bash
# Run inference on a single image
python scripts/inference.py \
    --model instruct_pix2pix \
    --checkpoint ./checkpoints/best \
    --original_image input.jpg \
    --sketch sketch.png \
    --prompt "add a red apple" \
    --output edited.png
```

## Dataset Structure

```
dataset/
├── sampled/           # Ground truth images + sketches (from Sketchy Database)
│   ├── 0/
│   │   ├── apple_n07739125_4340_photo.jpg    # Ground truth with object
│   │   └── apple_n07739125_4340-1_sketch.png # Object sketch
│   └── ...
├── inpainted/         # Images with objects removed (model inputs)
│   ├── 0/
│   │   └── apple_n07739125_4340_remove.jpg
│   └── ...
└── instructions/      # Text editing instructions
    ├── 0.txt
    └── ...
```

## Supported Models

| Model | Description | Status |
|-------|-------------|--------|
| `controlnet_scribble` | SD + ControlNet with scribble conditioning | ✅ Implemented |
| `instruct_pix2pix` | InstructPix2Pix for text-guided editing | ✅ Implemented |
| `controlnet_inpaint` | SD Inpaint + ControlNet Scribble | ✅ Implemented |
| `bagel` | ByteDance BAGEL unified multimodal model | 🔜 Planned |
| `t2i_adapter` | T2I-Adapter for controllable generation | 🔜 Planned |
