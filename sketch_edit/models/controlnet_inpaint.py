"""ControlNet + Inpainting model for sketch-guided image editing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps

from sketch_edit.models.base import BaseEditModel
from sketch_edit.utils import get_logger
from sketch_edit.utils.image import sketch_to_mask

logger = get_logger(__name__)


class ControlNetInpaintModel(BaseEditModel):
    """
    Stable Diffusion Inpainting with ControlNet Scribble conditioning.

    This model combines inpainting with ControlNet control, allowing
    sketch-guided editing of specific regions in an image.
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,  # float32 for CPU compatibility
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_id: str = "lllyasviel/sd-controlnet-scribble",
        vae_id: str | None = "stabilityai/sd-vae-ft-mse",
    ):
        """
        Initialize ControlNet Inpaint model.

        Args:
            device: Device to run on.
            dtype: Model dtype.
            model_id: Base Stable Diffusion model ID.
            controlnet_id: ControlNet model ID.
            vae_id: Optional VAE model ID for better quality.
        """
        super().__init__(device=device, dtype=dtype)
        self.model_id = model_id
        self.controlnet_id = controlnet_id
        self.vae_id = vae_id
        self.pipe = None
        self.controlnet = None

    def load(self, checkpoint_path: str | Path | None = None) -> None:
        """
        Load model weights.

        Args:
            checkpoint_path: Optional path to fine-tuned checkpoint.
        """
        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            StableDiffusionControlNetInpaintPipeline,
        )

        logger.info(f"Loading ControlNet from {self.controlnet_id}")
        self.controlnet = ControlNetModel.from_pretrained(
            self.controlnet_id,
            torch_dtype=self.dtype,
        )

        model_path = checkpoint_path or self.model_id
        logger.info(f"Loading Stable Diffusion from {model_path}")
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            model_path,
            controlnet=self.controlnet,
            torch_dtype=self.dtype,
            safety_checker=None,
        ).to(self.device)

        # Load better VAE if specified
        if self.vae_id:
            logger.info(f"Loading VAE from {self.vae_id}")
            vae = AutoencoderKL.from_pretrained(self.vae_id, torch_dtype=self.dtype)
            self.pipe.vae = vae

        self.enable_memory_optimizations()
        self._is_loaded = True
        logger.info("Model loaded successfully")

    def enable_memory_optimizations(self) -> None:
        """Enable memory optimizations."""
        if self.pipe is None:
            return

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            logger.info("xformers memory efficient attention enabled")
        except Exception:
            logger.warning("xformers not available")

    @torch.no_grad()
    def edit(
        self,
        original_image: Image.Image,
        sketch_image: Image.Image,
        prompt: str,
        mask_image: Image.Image | None = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        invert_sketch: bool = True,
        mask_thresh: int = 200,
        mask_kernel_size: int = 15,
        **kwargs: Any,
    ) -> Image.Image:
        """
        Edit an image using inpainting with ControlNet sketch control.

        Args:
            original_image: Original image to edit.
            sketch_image: Sketch/scribble control image.
            prompt: Text prompt describing the desired edit.
            mask_image: Optional mask image. If None, generated from sketch.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            controlnet_conditioning_scale: ControlNet conditioning strength.
            invert_sketch: Whether to invert the sketch colors.
            mask_thresh: Threshold for mask generation from sketch.
            mask_kernel_size: Kernel size for morphological closing.
            **kwargs: Additional arguments.

        Returns:
            Edited image.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Prepare images
        image = original_image.convert("RGB")
        sketch = sketch_image.convert("RGB")
        if invert_sketch:
            sketch = ImageOps.invert(sketch)

        # Generate mask from sketch if not provided
        if mask_image is None:
            # Save sketch temporarily to generate mask
            import tempfile
            import numpy as np

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                sketch_image.save(f.name)
                mask_array = sketch_to_mask(
                    f.name,
                    thresh_value=mask_thresh,
                    closing_kernel_size=mask_kernel_size,
                )
                mask_image = Image.fromarray(mask_array).convert("L")
        else:
            mask_image = mask_image.convert("L")

        # Generate
        result = self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            control_image=sketch,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )

        return result.images[0]

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        """Get trainable parameters (ControlNet parameters)."""
        if self.controlnet is None:
            return []
        return list(self.controlnet.parameters())

    def save(self, output_path: str | Path) -> None:
        """Save ControlNet checkpoint."""
        if self.controlnet is None:
            logger.warning("No ControlNet to save")
            return

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        self.controlnet.save_pretrained(output_path / "controlnet")
        logger.info(f"ControlNet saved to {output_path / 'controlnet'}")
