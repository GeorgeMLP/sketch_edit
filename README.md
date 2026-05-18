# ScribbleEdit: Synthetic Data for Image Editing with Scribbles and Text

A modular pipeline for multi-task synthetic dataset generation, model training, and evaluation for image editing guided by language and sketch. Paper link: https://arxiv.org/abs/2605.01135.

## Overview

This codebase provides tools for:
- **Multi-Task Dataset Generation**: Create synthetic datasets for sketch+text-guided image editing across multiple tasks
  - **Object Addition**: Remove object via inpainting; model adds it back
  - **Orientation Change**: Flip object horizontally; model restores correct orientation  
  - **Object Replacement**: Swap in foreign object; model replaces with correct one (using Insert-Anything)
  - **Object Completion**: Remove part of object at random granularity; model completes it (using UnSAMv2)
- **Model Training**: Fine-tune diffusion models for multimodal image editing
- **Evaluation**: Evaluate model predictions using multiple metrics (MSE, CLIP similarity, FID)
- **Train/Val/Test Splits**: Automatic 80/10/10 dataset splitting for proper evaluation

## Installation

```bash
# Create conda environment
conda create -n sketch_edit python=3.11 -y
conda activate sketch_edit

# Install the package in development mode
pip install -e ".[all]"
```

## Download Pretrained Models

Before generating the dataset, download the required models:

```bash
cd pretrained_models/
bash download_models.sh
cd ..
```

## Dataset Management

### Upload/Download Dataset (Wandb)

Upload your dataset to Weights & Biases for collaboration:

```bash
# Upload dataset
python scripts/wandb_sync.py \
    --action upload \
    --dataset_dir ./dataset \
    --artifact_name dataset

# Download dataset (automatically regenerates manifest with local paths)
python scripts/wandb_sync.py \
    --action download \
    --dataset_dir ./dataset \
    --artifact_name dataset:latest
```

## Quick Start

### 1. Generate Multi-Task Dataset

The pipeline now supports multiple editing tasks. Run the full pipeline with:

```bash
# Run all steps: segmentation → generate inputs → create manifest
python scripts/generate_dataset.py \
    --step all \
    --data_dir ./dataset \
    --num_samples 100  # Optional: limit number of samples

# Or run individual steps:

# Step 1: Segment objects using Grounding DINO + SAM
python scripts/generate_dataset.py \
    --step segmentation \
    --data_dir ./dataset

# Step 2: Generate task-specific inputs
# This creates input images for all tasks: object_addition, orientation_change, object_replacement, object_completion
python scripts/generate_dataset.py \
    --step generate \
    --data_dir ./dataset \
    --editing_tasks object_addition orientation_change object_replacement object_completion

# Step 3 (Optional): Generate VLM-based instructions using Gemma-3
python scripts/generate_dataset.py \
    --step instructions \
    --data_dir ./dataset \
    --editing_tasks object_addition orientation_change object_replacement object_completion

# Step 4: Create manifest with train/val/test splits
python scripts/generate_dataset.py \
    --step manifest \
    --data_dir ./dataset \
    --output_file manifest.json
```

### 2. Train a Model

Train on all tasks or filter by specific task:

```bash
# Train on all editing tasks
python scripts/train.py \
    --manifest dataset/manifest.json \
    --model controlnet_inpaint \
    --output_dir ./checkpoints \
    --batch_size 4 \
    --num_epochs 10

# Train on specific task only
python scripts/train.py \
    --manifest dataset/manifest.json \
    --task object_addition \
    --model controlnet_inpaint \
    --output_dir ./checkpoints/object_addition \
    --batch_size 4 \
    --num_epochs 10
```

### 3. Evaluate

Evaluate on test split with optional task filtering:

```bash
# Evaluate all tasks on test set
python scripts/evaluate.py \
    --manifest dataset/manifest.json \
    --split test \
    --predictions_dir ./predictions \
    --metrics mse hist_intersection clip_similarity \
    --output_file results.json

# Evaluate specific task only
python scripts/evaluate.py \
    --manifest dataset/manifest.json \
    --split test \
    --task object_addition \
    --predictions_dir ./predictions \
    --output_file results_obj_add.json
```

### 4. Inference

Run inference with task and split filtering:

```bash
# Single image inference
python scripts/inference.py \
    --model controlnet_inpaint \
    --checkpoint ./checkpoints/final \
    --original_image input.jpg \
    --sketch sketch.png \
    --prompt "Add a red apple to the scene" \
    --output edited.png

# Batch inference on test set
python scripts/inference.py \
    --model controlnet_inpaint \
    --checkpoint ./checkpoints/final \
    --manifest dataset/manifest.json \
    --split test \
    --output_dir ./predictions

# Batch inference on specific task
python scripts/inference.py \
    --model controlnet_inpaint \
    --checkpoint ./checkpoints/final \
    --manifest dataset/manifest.json \
    --split test \
    --task object_replacement \
    --output_dir ./predictions/obj_replacement
```

## Dataset Structure

```
dataset/
├── sampled/              # Ground truth images + sketches (from Sketchy Database)
│   ├── 0/
│   │   ├── apple_n07739125_4340_photo.jpg    # Ground truth with object
│   │   └── apple_n07739125_4340-1_sketch.png # Object sketch
│   └── ...
├── segmentation/         # Object segmentation results
│   ├── masks/
│   │   ├── 0_mask.png
│   │   └── ...
│   └── segmentation_results.json
├── generated/            # Task-specific generated inputs
│   ├── object_addition/
│   │   ├── 0/
│   │   │   └── input.jpg  # Background with object removed
│   │   ├── metadata.json
│   │   └── ...
│   ├── orientation_change/
│   │   ├── 0/
│   │   │   └── input.jpg  # Image with flipped object
│   │   └── ...
│   ├── object_replacement/
│   │   ├── 0/
│   │   │   └── input.jpg  # Image with foreign object inserted
│   │   └── ...
│   └── object_completion/
│       ├── 0/
│       │   └── input.jpg  # Image with partial object removed
│       └── ...
├── instructions/         # Optional VLM-generated instructions
│   ├── object_addition/
│   │   ├── 0.txt
│   │   └── ...
│   └── ...
├── splits.json          # Train/val/test split assignments (80/10/10)
└── dataset.json         # Multi-task manifest with splits
```

## Editing Tasks

| Task | Description | Implementation |
|------|-------------|----------------|
| **Object Addition** | Remove object via LaMa inpainting; model must add it back | Standard approach |
| **Orientation Change** | Flip object horizontally; model must restore correct orientation | Mask-based transformation |
| **Object Replacement** | Insert foreign object from different class; model must replace with correct one | **Insert-Anything (FLUX-based)** for natural composition |
| **Object Completion** | Remove part of object at random granularity; model must complete it | **UnSAMv2+** for granularity-aware partial segmentation |

**Note**: Object completion uses UnSAMv2+ by default (better performance than UnSAMv2). The checkpoint is loaded from `pretrained_models/unsamv2_plus_ckpt.pt`.

### Adding New Tasks

To add a new editing task:

1. Create a new class in `sketch_edit/data/generation/tasks/` inheriting from `EditingTask`
2. Implement `name`, `description`, and `generate_inputs()` method
3. Register it in `sketch_edit/data/generation/tasks/__init__.py`
4. The task will automatically be available in the pipeline

## Supported Models

| Model | Description | Status |
|-------|-------------|--------|
| `controlnet_scribble` | SD + ControlNet with scribble conditioning | ✅ Implemented |
| `instruct_pix2pix` | InstructPix2Pix for text-guided editing | ✅ Implemented |
| `controlnet_inpaint` | SD Inpaint + ControlNet Scribble | ✅ Implemented |
| `bagel` | ByteDance BAGEL unified multimodal model | 🔜 Planned |
| `t2i_adapter` | T2I-Adapter for controllable generation | 🔜 Planned |

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{ji2026scribbleedit,
    title={{ScribbleEdit: Synthetic Data for Image Editing with Scribbles and Text}},
    author={Ji, Anya and Ma, George and Wright, T{\'e}a and Zhang, Yiming and Chan, David M. and Suhr, Alane and Sojoudi, Somayeh},
    journal={arXiv preprint arXiv:2605.01135},
    year={2026}
}
```
