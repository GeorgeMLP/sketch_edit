"""Object removal via inpainting for dataset generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float, UInt8
from PIL import Image
from tqdm import tqdm

from sketch_edit.utils import get_logger, load_json

logger = get_logger(__name__)


class ObjectInpainter:
    """
    Remove objects from images via inpainting.

    Uses a diffusion-based inpainting model to fill in masked regions.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-inpainting",
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the inpainter.

        Args:
            model_id: HuggingFace model ID for inpainting model.
            device: Device to run model on.
            dtype: Model dtype.
        """
        from diffusers import StableDiffusionInpaintPipeline

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        logger.info(f"Loading inpainting model from {model_id}")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(self.device)

        # Enable memory optimizations
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            logger.warning("xformers not available, using default attention")

        logger.info("Inpainting model loaded successfully")

    @torch.no_grad()
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "empty background, natural scene",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> Image.Image:
        """
        Inpaint (remove object from) an image using the provided mask.

        Args:
            image: Input image with the object.
            mask: Binary mask where white (255) indicates the region to inpaint.
            prompt: Text prompt guiding the inpainting.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.

        Returns:
            Inpainted image with object removed.
        """
        image_rgb = image.convert("RGB")
        mask_l = mask.convert("L")

        result = self.pipe(
            prompt=prompt,
            image=image_rgb,
            mask_image=mask_l,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        return result.images[0]

    def inpaint_dataset(
        self,
        data_dir: str | Path,
        segmentation_dir: str | Path,
        output_dir: str | Path,
        prompt: str = "empty natural background",
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
            prompt: Text prompt for inpainting.
            skip_existing: Skip samples that already have inpainted images.
        """
        data_dir = Path(data_dir)
        segmentation_dir = Path(segmentation_dir)
        output_dir = Path(output_dir)

        sampled_dir = data_dir / "sampled"
        mask_dir = segmentation_dir / "masks"

        # Find all sample directories
        sample_ids = sorted([d.name for d in sampled_dir.iterdir() if d.is_dir()])
        logger.info(f"Processing {len(sample_ids)} samples")

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

                inpainted = self.inpaint(image, mask, prompt=prompt)
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
