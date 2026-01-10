"""ControlNet Scribble model for sketch-guided image generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from jaxtyping import Float
from PIL import Image, ImageOps

from sketch_edit.models.base import BaseEditModel
from sketch_edit.utils import get_logger

logger = get_logger(__name__)


class ControlNetScribbleModel(BaseEditModel):
    """
    Stable Diffusion with ControlNet Scribble conditioning.

    This model generates images conditioned on scribble/sketch inputs.
    It does NOT use the original image, only the sketch and text prompt.
    Suitable for sketch-to-image generation.
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float16,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_id: str = "lllyasviel/sd-controlnet-scribble",
    ):
        """
        Initialize ControlNet Scribble model.

        Args:
            device: Device to run on.
            dtype: Model dtype.
            model_id: Base Stable Diffusion model ID.
            controlnet_id: ControlNet model ID.
        """
        super().__init__(device=device, dtype=dtype)
        self.model_id = model_id
        self.controlnet_id = controlnet_id
        self.pipe = None
        self.controlnet = None

    def load(self, checkpoint_path: str | Path | None = None) -> None:
        """
        Load model weights.

        Args:
            checkpoint_path: Optional path to fine-tuned checkpoint.
        """
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetPipeline,
            UniPCMultistepScheduler,
        )

        logger.info(f"Loading ControlNet from {self.controlnet_id}")
        self.controlnet = ControlNetModel.from_pretrained(
            self.controlnet_id,
            torch_dtype=self.dtype,
        )

        model_path = checkpoint_path or self.model_id
        logger.info(f"Loading Stable Diffusion from {model_path}")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_path,
            controlnet=self.controlnet,
            torch_dtype=self.dtype,
            safety_checker=None,
        ).to(self.device)

        # Use efficient scheduler
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

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

        try:
            self.pipe.enable_model_cpu_offload()
        except Exception:
            pass

    @torch.no_grad()
    def edit(
        self,
        original_image: Image.Image,
        sketch_image: Image.Image,
        prompt: str,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        invert_sketch: bool = True,
        **kwargs: Any,
    ) -> Image.Image:
        """
        Generate an image from sketch and text prompt.

        Note: This model ignores the original_image parameter since it
        generates from scratch based on the sketch.

        Args:
            original_image: Original image (ignored by this model).
            sketch_image: Sketch/scribble control image.
            prompt: Text prompt describing the desired output.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            controlnet_conditioning_scale: ControlNet conditioning strength.
            invert_sketch: Whether to invert the sketch colors.
            **kwargs: Additional arguments.

        Returns:
            Generated image.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Prepare sketch image
        sketch = sketch_image.convert("RGB")
        if invert_sketch:
            sketch = ImageOps.invert(sketch)

        # Generate
        result = self.pipe(
            prompt=prompt,
            image=sketch,
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
