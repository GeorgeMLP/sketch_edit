"""InstructPix2Pix model for text-guided image editing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from jaxtyping import Float
from PIL import Image

from sketch_edit.models.base import BaseEditModel
from sketch_edit.utils import get_logger

logger = get_logger(__name__)


class InstructPix2PixModel(BaseEditModel):
    """
    InstructPix2Pix model for text-guided image editing.

    This model edits images based on natural language instructions.
    The sketch input is currently not used but can be integrated
    via additional conditioning in future versions.
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float16,
        model_id: str = "timbrooks/instruct-pix2pix",
    ):
        """
        Initialize InstructPix2Pix model.

        Args:
            device: Device to run on.
            dtype: Model dtype.
            model_id: Model ID or path to fine-tuned checkpoint.
        """
        super().__init__(device=device, dtype=dtype)
        self.model_id = model_id
        self.pipe = None

    def load(self, checkpoint_path: str | Path | None = None) -> None:
        """
        Load model weights.

        Args:
            checkpoint_path: Optional path to fine-tuned checkpoint.
        """
        from diffusers import StableDiffusionInstructPix2PixPipeline

        model_path = checkpoint_path or self.model_id
        logger.info(f"Loading InstructPix2Pix from {model_path}")

        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            safety_checker=None,
        ).to(self.device)

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
        num_inference_steps: int = 100,
        image_guidance_scale: float = 1.5,
        guidance_scale: float = 7.5,
        **kwargs: Any,
    ) -> Image.Image:
        """
        Edit an image based on text instruction.

        Note: The sketch_image is currently not used by this model.
        For sketch-aware editing, consider using ControlNetInpaintModel.

        Args:
            original_image: Original image to edit.
            sketch_image: Sketch image (currently unused).
            prompt: Text instruction for the edit.
            num_inference_steps: Number of denoising steps.
            image_guidance_scale: How much to follow the original image.
            guidance_scale: Classifier-free guidance scale.
            **kwargs: Additional arguments.

        Returns:
            Edited image.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        image = original_image.convert("RGB")

        result = self.pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
        )

        return result.images[0]

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        """Get trainable parameters (UNet parameters)."""
        if self.pipe is None:
            return []
        return list(self.pipe.unet.parameters())

    def save(self, output_path: str | Path) -> None:
        """Save model checkpoint."""
        if self.pipe is None:
            logger.warning("No pipeline to save")
            return

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        self.pipe.save_pretrained(output_path)
        logger.info(f"Model saved to {output_path}")


class InstructPix2PixWithSketchModel(BaseEditModel):
    """
    InstructPix2Pix with additional sketch conditioning.

    This extends InstructPix2Pix by concatenating the original image
    latent with sketch information for multi-modal conditioning.
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float16,
        base_model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_id: str = "lllyasviel/control_v11p_sd15_scribble",
    ):
        """
        Initialize InstructPix2Pix with sketch conditioning.

        Args:
            device: Device to run on.
            dtype: Model dtype.
            base_model_id: Base Stable Diffusion model ID.
            controlnet_id: ControlNet model ID for sketch conditioning.
        """
        super().__init__(device=device, dtype=dtype)
        self.base_model_id = base_model_id
        self.controlnet_id = controlnet_id

        self.vae = None
        self.unet = None
        self.controlnet = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None

    def load(self, checkpoint_path: str | Path | None = None) -> None:
        """Load model components."""
        from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        logger.info("Loading model components...")

        # Load tokenizer and text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.base_model_id, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.base_model_id, subfolder="text_encoder"
        ).to(self.device)
        self.text_encoder.requires_grad_(False)

        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.base_model_id, subfolder="vae"
        ).to(self.device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        # Load UNet (modify input channels for concatenated conditioning)
        unet_path = checkpoint_path if checkpoint_path else self.base_model_id
        self.unet = UNet2DConditionModel.from_pretrained(
            unet_path, subfolder="unet" if not checkpoint_path else None
        ).to(self.device)

        # Modify UNet input channels if needed (8 channels: 4 noisy + 4 original)
        if self.unet.config.in_channels == 4:
            self._expand_unet_input_channels()

        # Load ControlNet
        self.controlnet = ControlNetModel.from_pretrained(self.controlnet_id).to(self.device)

        # Load scheduler
        self.scheduler = DDPMScheduler.from_pretrained(self.base_model_id, subfolder="scheduler")

        # Enable optimizations
        self.unet.enable_gradient_checkpointing()
        self.controlnet.enable_gradient_checkpointing()

        try:
            self.unet.enable_xformers_memory_efficient_attention()
            self.controlnet.enable_xformers_memory_efficient_attention()
        except Exception:
            logger.warning("xformers not available")

        self._is_loaded = True
        logger.info("Model loaded successfully")

    def _expand_unet_input_channels(self) -> None:
        """Expand UNet input channels from 4 to 8."""
        import torch.nn as nn

        old_conv = self.unet.conv_in
        new_conv = nn.Conv2d(
            8,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
            device=self.device,
        )

        with torch.no_grad():
            new_conv.weight.zero_()
            new_conv.weight[:, :4] = old_conv.weight
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        self.unet.conv_in = new_conv
        self.unet.register_to_config(in_channels=8)

    @torch.no_grad()
    def edit(
        self,
        original_image: Image.Image,
        sketch_image: Image.Image,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        **kwargs: Any,
    ) -> Image.Image:
        """Edit an image using text + sketch conditioning."""
        raise NotImplementedError(
            "Full inference not yet implemented. "
            "Use training pipeline or InstructPix2PixModel for inference."
        )

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        """Get trainable parameters (UNet new conv + ControlNet)."""
        params = []
        if self.unet is not None:
            params.extend(list(self.unet.conv_in.parameters()))
            params.extend([p for n, p in self.unet.named_parameters() if "conv_in" not in n])
        if self.controlnet is not None:
            params.extend(list(self.controlnet.parameters()))
        return params

    def save(self, output_path: str | Path) -> None:
        """Save model checkpoints."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.unet is not None:
            self.unet.save_pretrained(output_path / "unet")
        if self.controlnet is not None:
            self.controlnet.save_pretrained(output_path / "controlnet")

        logger.info(f"Model saved to {output_path}")
