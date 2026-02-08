"""Utilities for natural object insertion using FLUX-based diffusion (Insert-Anything).

This module provides a minimal extraction of the Insert-Anything project for compositing
objects naturally into scenes. Only inference-time components are included.

Reference: https://github.com/song-wensong/insert-anything
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline
from jaxtyping import UInt8
from PIL import Image

from sketch_edit.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Geometric utilities (from insert-anything/utils/utils.py)
# ---------------------------------------------------------------------------

def _adaptive_ratio(r: float, T: float = 0.6, beta: float = 0.1) -> float:
    """Adaptive scaling ratio based on object size."""
    return np.where(r < T, beta + (1 - beta) / T * r, 1)


def get_bbox_from_mask(mask: UInt8[np.ndarray, "H W"]) -> tuple[int, int, int, int]:
    """Extract bounding box (y1, y2, x1, x2) from a binary mask."""
    h, w = mask.shape[:2]
    if mask.sum() < 10:
        return 0, h, 0, w
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return int(y1), int(y2), int(x1), int(x2)


def expand_bbox(
    image: UInt8[np.ndarray, "H W ..."],
    yyxx: tuple[int, int, int, int],
    ratio: float,
) -> tuple[int, int, int, int]:
    """Expand bounding box adaptively based on object size."""
    y1, y2, x1, x2 = yyxx
    H, W = image.shape[:2]

    yyxx_area = (y2 - y1 + 1) * (x2 - x1 + 1)
    r1 = yyxx_area / (H * W)
    r2 = _adaptive_ratio(r1)
    ratio = float(np.sqrt(r2 / r1))

    xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    h = ratio * (y2 - y1 + 1)
    w = ratio * (x2 - x1 + 1)

    x1 = int(xc - w * 0.5)
    x2 = int(xc + w * 0.5)
    y1 = int(yc - h * 0.5)
    y2 = int(yc + h * 0.5)

    return max(0, y1), min(H, y2), max(0, x1), min(W, x2)


def box2square(
    image: UInt8[np.ndarray, "H W ..."],
    box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Convert bounding box to square shape."""
    H, W = image.shape[:2]
    y1, y2, x1, x2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    h, w = y2 - y1, x2 - x1

    if h >= w:
        x1 = cx - h // 2
        x2 = cx + h // 2
    else:
        y1 = cy - w // 2
        y2 = cy + w // 2

    return max(0, y1), min(H, y2), max(0, x1), min(W, x2)


def pad_to_square(
    image: UInt8[np.ndarray, "H W ..."],
    pad_value: int = 255,
) -> UInt8[np.ndarray, "S S ..."]:
    """Pad image to square shape (center padding)."""
    H, W = image.shape[:2]
    if H == W:
        return image

    padd = abs(H - W)
    padd_1 = padd // 2
    padd_2 = padd - padd_1

    if len(image.shape) == 2:
        pad_param = ((0, 0), (padd_1, padd_2)) if H > W else ((padd_1, padd_2), (0, 0))
    else:
        pad_param = ((0, 0), (padd_1, padd_2), (0, 0)) if H > W else ((padd_1, padd_2), (0, 0), (0, 0))

    return np.pad(image, pad_param, "constant", constant_values=pad_value)


def expand_image_mask(
    image: UInt8[np.ndarray, "H W 3"],
    mask: UInt8[np.ndarray, "H W"],
    ratio: float = 1.4,
) -> tuple[UInt8[np.ndarray, "Hn Wn 3"], UInt8[np.ndarray, "Hn Wn"]]:
    """Expand image and mask by padding with white/zeros."""
    h, w = image.shape[:2]
    H, W = int(h * ratio), int(w * ratio)
    h1 = (H - h) // 2
    h2 = H - h - h1
    w1 = (W - w) // 2
    w2 = W - w - w1

    image = np.pad(image, ((h1, h2), (w1, w2), (0, 0)), "constant", constant_values=255)
    mask = np.pad(mask, ((h1, h2), (w1, w2)), "constant", constant_values=0)
    return image, mask


def crop_back(
    pred: UInt8[np.ndarray, "S S 3"],
    tar_image: UInt8[np.ndarray, "H W 3"],
    extra_sizes: tuple[int, int, int, int],
    tar_box_yyxx_crop: tuple[int, int, int, int],
) -> UInt8[np.ndarray, "H W 3"]:
    """Crop predicted square image back to original dimensions and paste into target."""
    H1, W1, H2, W2 = extra_sizes
    y1, y2, x1, x2 = tar_box_yyxx_crop
    pred = cv2.resize(pred, (W2, H2))
    m = 2  # margin pixels

    if W1 == H1:
        if m != 0:
            tar_image[y1 + m : y2 - m, x1 + m : x2 - m, :] = pred[m:-m, m:-m]
        else:
            tar_image[y1:y2, x1:x2, :] = pred[:, :]
        return tar_image

    # Unpad to original aspect ratio
    if W1 < W2:
        pad1 = (W2 - W1) // 2
        pad2 = W2 - W1 - pad1
        pred = pred[:, pad1:-pad2, :]
    else:
        pad1 = (H2 - H1) // 2
        pad2 = H2 - H1 - pad1
        pred = pred[pad1:-pad2, :, :]

    gen_image = tar_image.copy()
    if m != 0:
        gen_image[y1 + m : y2 - m, x1 + m : x2 - m, :] = pred[m:-m, m:-m]
    else:
        gen_image[y1:y2, x1:x2, :] = pred[:, :]

    return gen_image


# ---------------------------------------------------------------------------
# FLUX Redux image embedding (from insert-anything/src/models/image_project.py)
# ---------------------------------------------------------------------------

def _encode_reference_image(
    redux_pipeline: FluxPriorReduxPipeline,
    image: Image.Image,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract FLUX Redux embeddings from reference image."""
    batch_size = 1

    # Encode image to latents and extract embeddings
    image_latents = redux_pipeline.encode_image(image, device, 1)
    image_embeds = redux_pipeline.image_embedder(image_latents).image_embeds.to(device)

    # Create zero text embeddings (image-only conditioning)
    dtype = image_embeds.dtype
    prompt_embeds = torch.zeros((batch_size, 512, 4096), device=device, dtype=dtype)
    pooled_prompt_embeds = torch.zeros((batch_size, 768), device=device, dtype=dtype)

    # Concatenate image embeddings
    prompt_embeds = torch.cat([prompt_embeds, image_embeds], dim=1)

    return prompt_embeds, pooled_prompt_embeds


# ---------------------------------------------------------------------------
# Main compositor class
# ---------------------------------------------------------------------------

class InsertAnythingCompositor:
    """Natural object insertion using FLUX.1-Fill + Redux with LoRA.

    This wraps the Insert-Anything inference workflow for compositing objects
    naturally into scenes with proper lighting, shadows, and perspective.
    """

    def __init__(
        self,
        lora_path: str | Path,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        size: tuple[int, int] = (768, 768),
        enable_cpu_offload: bool = False,
        enable_vae_slicing: bool = False,
    ):
        """Initialize the compositor with FLUX models and LoRA weights.

        Args:
            lora_path: Path to Insert-Anything LoRA checkpoint (.safetensors).
            device: Device to run models on.
            dtype: Model dtype (bfloat16 recommended).
            size: Processing resolution (default 768x768 as in Insert-Anything).
            enable_cpu_offload: Enable CPU offloading to reduce VRAM (~10GB vs 24GB).
            enable_vae_slicing: Enable VAE slicing to reduce memory spikes.
        """
        self.device = torch.device(device)
        self.dtype = dtype
        self.size = size

        logger.info(f"Loading FLUX.1-Fill-dev pipeline on {self.device}...")
        self.fill_pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            torch_dtype=dtype,
        ).to(self.device)

        logger.info(f"Loading LoRA weights from {lora_path}...")
        self.fill_pipe.load_lora_weights(str(lora_path))

        logger.info(f"Loading FLUX.1-Redux-dev pipeline on {self.device}...")
        self.redux_pipe = FluxPriorReduxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Redux-dev",
            torch_dtype=dtype,
        ).to(self.device)

        # Apply memory optimizations
        if enable_cpu_offload:
            logger.info("Enabling CPU offloading (reduces VRAM to ~10-12GB)")
            self.fill_pipe.enable_model_cpu_offload()
            self.redux_pipe.enable_model_cpu_offload()
        
        if enable_vae_slicing:
            logger.info("Enabling VAE slicing (reduces memory spikes)")
            self.fill_pipe.enable_vae_slicing()

        logger.info("Insert-Anything compositor ready")

    def composite(
        self,
        ref_image: UInt8[np.ndarray, "H W 3"],
        ref_mask: UInt8[np.ndarray, "H W"],
        tar_image: UInt8[np.ndarray, "H W 3"],
        tar_mask: UInt8[np.ndarray, "H W"],
        seed: int = 666,
        num_inference_steps: int = 10,
    ) -> UInt8[np.ndarray, "H W 3"]:
        """Composite reference object into target scene naturally.

        Args:
            ref_image: Reference image containing the object (RGB, 0-255).
            ref_mask: Binary mask of reference object (0 or 255).
            tar_image: Target scene image (RGB, 0-255).
            tar_mask: Binary mask indicating where to place object (0 or 255).
            seed: Random seed for reproducibility.
            num_inference_steps: Number of denoising steps (default 10).
                - 50 steps: highest quality, ~40s/sample
                - 20-25 steps: good quality, ~16-20s/sample (recommended)
                - 15 steps: acceptable quality, ~12s/sample
                - 10 steps: lower quality, ~8s/sample

        Returns:
            Composited image with reference object naturally inserted (RGB, 0-255).
        """
        # Prepare reference object (masked, cropped, centered on white background)
        ref_box = get_bbox_from_mask(ref_mask)
        ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
        masked_ref = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1 - ref_mask_3)

        y1, y2, x1, x2 = ref_box
        masked_ref = masked_ref[y1:y2, x1:x2, :]
        ref_mask_crop = ref_mask[y1:y2, x1:x2]
        masked_ref, ref_mask_crop = expand_image_mask(masked_ref, ref_mask_crop, ratio=1.3)
        masked_ref = pad_to_square(masked_ref, pad_value=255)
        masked_ref = cv2.resize(masked_ref.astype(np.uint8), self.size).astype(np.uint8)

        # Dilate target mask
        kernel = np.ones((7, 7), np.uint8)
        tar_mask = cv2.dilate(tar_mask, kernel, iterations=2)

        # Zoom into target region
        tar_box = get_bbox_from_mask(tar_mask)
        tar_box = expand_bbox(tar_mask, tar_box, ratio=1.2)
        tar_box_crop = expand_bbox(tar_image, tar_box, ratio=2)
        tar_box_crop = box2square(tar_image, tar_box_crop)
        y1, y2, x1, x2 = tar_box_crop

        old_tar_image = tar_image.copy()
        tar_image_crop = tar_image[y1:y2, x1:x2, :]
        tar_mask_crop = tar_mask[y1:y2, x1:x2]

        H1, W1 = tar_image_crop.shape[:2]
        tar_mask_crop = pad_to_square(tar_mask_crop, pad_value=0)
        tar_mask_crop = cv2.resize(tar_mask_crop, self.size)

        # Extract reference embeddings
        pipe_prior_output = _encode_reference_image(
            self.redux_pipe,
            Image.fromarray(masked_ref),
            self.device,
        )
        prompt_embeds, pooled_prompt_embeds = pipe_prior_output

        # Prepare target image
        tar_image_crop = pad_to_square(tar_image_crop, pad_value=255)
        H2, W2 = tar_image_crop.shape[:2]
        tar_image_crop = cv2.resize(tar_image_crop, self.size)

        # Create diptych (ref | tar) for FLUX Fill
        diptych = np.concatenate([masked_ref, tar_image_crop], axis=1)
        tar_mask_3 = np.stack([tar_mask_crop, tar_mask_crop, tar_mask_crop], -1)
        mask_black = np.zeros_like(tar_image_crop)
        mask_diptych = np.concatenate([mask_black, tar_mask_3], axis=1)

        diptych_pil = Image.fromarray(diptych)
        mask_diptych[mask_diptych == 1] = 255
        mask_diptych_pil = Image.fromarray(mask_diptych)

        # Run FLUX Fill with LoRA
        generator = torch.Generator(self.device).manual_seed(seed)
        edited = self.fill_pipe(
            image=diptych_pil,
            mask_image=mask_diptych_pil,
            height=mask_diptych_pil.size[1],
            width=mask_diptych_pil.size[0],
            max_sequence_length=512,
            num_inference_steps=num_inference_steps,
            generator=generator,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        ).images[0]

        # Extract right half (target result)
        width, height = edited.size
        edited = edited.crop((width // 2, 0, width, height))

        # Crop back to original size
        edited = np.array(edited)
        edited = crop_back(edited, old_tar_image, (H1, W1, H2, W2), tar_box_crop)

        return edited
