"""Object removal via inpainting using LaMa model."""

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
from tqdm import tqdm

from sketch_edit.saicinpainting.training.trainers import load_checkpoint
from sketch_edit.utils import get_logger

logging.getLogger("sketch_edit.saicinpainting").setLevel(logging.ERROR)
logging.getLogger("root").setLevel(logging.WARNING)

logger = get_logger(__name__)


def dilate_mask(
    mask: UInt8[np.ndarray, "H W"],
    kernel_size: int = 15,
) -> UInt8[np.ndarray, "H W"]:
    """
    Dilate a binary mask to expand the inpainting region.
    
    This helps avoid edge artifacts by ensuring the inpainting model
    has enough context around the object boundary.
    
    Args:
        mask: Binary mask array (H, W) with values 0-255.
        kernel_size: Size of the dilation kernel.
        
    Returns:
        Dilated mask array (H, W) with values 0-255.
    """
    mask = mask.astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    return dilated


def _pad_tensor_to_modulo(
    tensor: Float[torch.Tensor, "B C H W"],
    mod: int,
) -> Float[torch.Tensor, "B C Hp Wp"]:
    """Pad tensor dimensions to be divisible by mod."""
    _, _, h, w = tensor.shape
    pad_h = (mod - h % mod) % mod
    pad_w = (mod - w % mod) % mod
    if pad_h == 0 and pad_w == 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")


def _move_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensors in batch to device."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


class LamaInpainter:
    """
    Remove objects from images via inpainting using LaMa model.

    LaMa (Large Mask Inpainting) is specifically designed for high-resolution
    inpainting and works well for object removal without text prompts.
    """

    def __init__(
        self,
        lama_ckpt_dir: str | Path = "pretrained_models/big-lama",
        lama_config_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ):
        """
        Initialize the LaMa inpainter.

        Args:
            lama_ckpt_dir: Path to the LaMa checkpoint directory (containing config.yaml and models/).
            lama_config_path: Optional path to prediction config. If None, uses defaults.
            device: Device to run model on.
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.lama_ckpt_dir = Path(lama_ckpt_dir)
        
        # Set up paths
        train_config_path = self.lama_ckpt_dir / "config.yaml"
        checkpoint_path = self.lama_ckpt_dir / "models" / "best.ckpt"
        
        if not train_config_path.exists():
            raise FileNotFoundError(f"LaMa config not found: {train_config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"LaMa checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading LaMa model from {self.lama_ckpt_dir}")
        
        # Load training config
        with open(train_config_path, "r") as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = "noop"
        
        # Load model
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
        """
        Inpaint (remove object from) an image using the provided mask.

        Args:
            image: Input image with the object (PIL Image).
            mask: Binary mask where white (255) indicates the region to inpaint.
            dilate_kernel_size: Kernel size for mask dilation. Set to 0 to disable.
            mod: Modulo for padding (LaMa requires dimensions divisible by 8).

        Returns:
            Inpainted image with object removed.
        """
        # Convert to numpy arrays
        img_np: UInt8[np.ndarray, "H W 3"] = np.array(image.convert("RGB"))
        mask_np: UInt8[np.ndarray, "H W"] = np.array(mask.convert("L"))
        
        # Dilate mask to avoid edge artifacts
        if dilate_kernel_size > 0:
            mask_np = dilate_mask(mask_np, kernel_size=dilate_kernel_size)
        
        # Ensure mask is binary (0 or 255)
        if np.max(mask_np) == 1:
            mask_np = mask_np * 255
        
        # Convert to tensors
        img_tensor: Float[torch.Tensor, "H W 3"] = torch.from_numpy(img_np).float().div(255.0)
        mask_tensor: Float[torch.Tensor, "H W"] = torch.from_numpy(mask_np).float()
        
        # Prepare batch
        batch = {
            "image": img_tensor.permute(2, 0, 1).unsqueeze(0),  # (1, 3, H, W)
            "mask": mask_tensor[None, None],  # (1, 1, H, W)
        }
        
        # Store original size for unpadding
        orig_height, orig_width = batch["image"].shape[2], batch["image"].shape[3]
        
        # Pad to modulo
        batch["image"] = _pad_tensor_to_modulo(batch["image"], mod)
        batch["mask"] = _pad_tensor_to_modulo(batch["mask"], mod)
        
        # Move to device and binarize mask
        batch = _move_to_device(batch, self.device)
        batch["mask"] = (batch["mask"] > 0).float()
        
        # Run inpainting
        batch = self.model(batch)
        
        # Get result
        result: Float[torch.Tensor, "3 Hp Wp"] = batch["inpainted"][0]
        result = result.permute(1, 2, 0)  # (Hp, Wp, 3)
        result = result.detach().cpu().numpy()
        
        # Unpad to original size
        result = result[:orig_height, :orig_width]
        
        # Convert back to image
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result)

    def inpaint_dataset(
        self,
        data_dir: str | Path,
        segmentation_dir: str | Path,
        output_dir: str | Path,
        dilate_kernel_size: int = 15,
        skip_existing: bool = True,
    ) -> None:
        """
        Remove objects from all images in a dataset directory.

        Expects:
        - data_dir/sampled/{id}/*_photo.jpg (images with objects)
        - segmentation_dir/masks/{id}_mask.png (object masks)

        Args:
            data_dir: Root directory containing sampled/.
            segmentation_dir: Directory containing segmentation masks.
            output_dir: Directory to save inpainted images.
            dilate_kernel_size: Kernel size for mask dilation (default: 15, like Inpaint-Anything).
            skip_existing: Skip samples that already have inpainted images.
        """
        data_dir = Path(data_dir)
        segmentation_dir = Path(segmentation_dir)
        output_dir = Path(output_dir)

        sampled_dir = data_dir / "sampled"
        mask_dir = segmentation_dir / "masks"

        # Find all sample directories
        sample_ids = sorted([d.name for d in sampled_dir.iterdir() if d.is_dir()])
        logger.info(f"Processing {len(sample_ids)} samples with LaMa inpainting")

        for sample_id in tqdm(sample_ids, desc="Inpainting objects"):
            sample_output_dir = output_dir / sample_id
            sample_output_dir.mkdir(parents=True, exist_ok=True)

            # Find existing inpainted images
            existing_outputs = list(sample_output_dir.glob("*_remove.jpg"))
            if skip_existing and existing_outputs:
                continue

            # Find source image and mask
            photo_files = list((sampled_dir / sample_id).glob("*_photo.jpg"))
            mask_path = mask_dir / f"{sample_id}_mask.png"

            if not photo_files or not mask_path.exists():
                logger.warning(f"Missing image or mask for sample {sample_id}")
                continue

            photo_path = photo_files[0]
            output_path = sample_output_dir / photo_path.name.replace("_photo.jpg", "_remove.jpg")

            try:
                image = Image.open(photo_path).convert("RGB")
                mask = Image.open(mask_path).convert("L")

                inpainted = self.inpaint(
                    image, 
                    mask, 
                    dilate_kernel_size=dilate_kernel_size,
                )
                inpainted.save(output_path)

            except Exception as e:
                logger.error(f"Failed to inpaint sample {sample_id}: {e}")

        logger.info(f"Inpainting complete. Results saved to {output_dir}")


def copy_existing_inpainted(
    source_dir: str | Path,
    target_dir: str | Path,
) -> None:
    """
    Copy existing inpainted images to a new location.

    Useful when you already have pre-generated inpainted images and
    want to use them without regenerating.

    Args:
        source_dir: Directory containing existing inpainted images.
        target_dir: Directory to copy images to.
    """
    import shutil

    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    if not source_dir.exists():
        logger.warning(f"Source directory does not exist: {source_dir}")
        return

    sample_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    logger.info(f"Copying {len(sample_dirs)} samples from {source_dir} to {target_dir}")

    for sample_dir in tqdm(sample_dirs, desc="Copying inpainted images"):
        target_sample_dir = target_dir / sample_dir.name
        if target_sample_dir.exists():
            continue

        shutil.copytree(sample_dir, target_sample_dir)

    logger.info("Copy complete")
