"""LaMa inpainting model for object removal."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="lightning_fabric")

import cv2
import numpy as np
import torch
import yaml
from jaxtyping import Float, UInt8
from omegaconf import OmegaConf
from PIL import Image

from sketch_edit.saicinpainting.training.trainers import load_checkpoint
from sketch_edit.utils import get_logger

logging.getLogger("sketch_edit.saicinpainting").setLevel(logging.ERROR)
logging.getLogger("root").setLevel(logging.WARNING)

logger = get_logger(__name__)


def dilate_mask(
    mask: UInt8[np.ndarray, "H W"],
    kernel_size: int = 15,
) -> UInt8[np.ndarray, "H W"]:
    """Dilate a binary mask to expand the inpainting region."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def _pad_to_modulo(
    tensor: Float[torch.Tensor, "B C H W"], mod: int,
) -> Float[torch.Tensor, "B C Hp Wp"]:
    _, _, h, w = tensor.shape
    pad_h = (mod - h % mod) % mod
    pad_w = (mod - w % mod) % mod
    if pad_h == 0 and pad_w == 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")


class LamaInpainter:
    """Remove objects from images via LaMa inpainting."""

    def __init__(
        self,
        lama_ckpt_dir: str | Path = "pretrained_models/big-lama",
        device: str | torch.device | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        lama_ckpt_dir = Path(lama_ckpt_dir)

        train_config_path = lama_ckpt_dir / "config.yaml"
        checkpoint_path = lama_ckpt_dir / "models" / "best.ckpt"
        if not train_config_path.exists():
            raise FileNotFoundError(f"LaMa config not found: {train_config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"LaMa checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading LaMa model from {lama_ckpt_dir}")
        with open(train_config_path) as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = "noop"

        self.model = load_checkpoint(train_config, str(checkpoint_path), strict=False, map_location="cpu")
        self.model.freeze()
        self.model.to(self.device)
        logger.info("LaMa model loaded successfully")

    @torch.no_grad()
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        dilate_kernel_size: int = 15,
        mod: int = 8,
    ) -> Image.Image:
        """Inpaint the masked region, returning the image with the object removed."""
        img_np: UInt8[np.ndarray, "H W 3"] = np.array(image.convert("RGB"))
        mask_np: UInt8[np.ndarray, "H W"] = np.array(mask.convert("L"))

        if dilate_kernel_size > 0:
            mask_np = dilate_mask(mask_np, kernel_size=dilate_kernel_size)
        if np.max(mask_np) == 1:
            mask_np = mask_np * 255

        img_t = torch.from_numpy(img_np).float().div(255.0).permute(2, 0, 1).unsqueeze(0)
        mask_t = torch.from_numpy(mask_np).float()[None, None]

        orig_h, orig_w = img_t.shape[2], img_t.shape[3]
        batch = {
            "image": _pad_to_modulo(img_t, mod),
            "mask": _pad_to_modulo(mask_t, mod),
        }
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch["mask"] = (batch["mask"] > 0).float()

        batch = self.model(batch)

        result = batch["inpainted"][0].permute(1, 2, 0).cpu().numpy()
        result = result[:orig_h, :orig_w]
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result)
