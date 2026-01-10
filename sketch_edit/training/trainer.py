"""Trainer for sketch-guided image editing models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from jaxtyping import Float, Int
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from sketch_edit.data.transforms import normalize_for_diffusion
from sketch_edit.training.losses import DiffusionLoss
from sketch_edit.utils import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Model
    model_name: str = "controlnet_inpaint"
    pretrained_model_id: str = "runwayml/stable-diffusion-v1-5"
    controlnet_id: str = "lllyasviel/control_v11p_sd15_scribble"

    # Training
    batch_size: int = 4
    num_epochs: int = 10
    max_steps: int | None = None
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    conditioning_dropout_prob: float = 0.1

    # Data
    image_size: int = 512
    val_split: float = 0.1

    # Checkpointing
    output_dir: str = "./checkpoints"
    save_every_n_steps: int = 500
    eval_every_n_steps: int = 100
    log_every_n_steps: int = 10

    # Hardware
    mixed_precision: bool = True
    device: str = "cuda"
    seed: int = 42


class Trainer:
    """
    Trainer for sketch-guided image editing models.

    Supports training with ControlNet conditioning and text guidance.
    """

    def __init__(
        self,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration.
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(config.device)

        # Initialize models
        self._setup_models()
        self._setup_optimizer()
        self._setup_loss()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None

    def _setup_models(self) -> None:
        """Set up model components."""
        from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        logger.info("Loading model components...")

        # Tokenizer and text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.pretrained_model_id, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.pretrained_model_id, subfolder="text_encoder"
        ).to(self.device)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

        # VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.config.pretrained_model_id, subfolder="vae"
        ).to(self.device)
        self.vae.requires_grad_(False)
        self.vae.eval()

        # UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_model_id, subfolder="unet"
        ).to(self.device)

        # Expand UNet input channels for InstructPix2Pix-style conditioning
        self._expand_unet_channels()

        # ControlNet
        self.controlnet = ControlNetModel.from_pretrained(self.config.controlnet_id).to(self.device)

        # Scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.config.pretrained_model_id, subfolder="scheduler"
        )

        # Enable memory optimizations
        self.unet.enable_gradient_checkpointing()
        self.controlnet.enable_gradient_checkpointing()

        try:
            self.unet.enable_xformers_memory_efficient_attention()
            self.controlnet.enable_xformers_memory_efficient_attention()
        except Exception:
            logger.warning("xformers not available")

        logger.info("Models loaded")

    def _expand_unet_channels(self) -> None:
        """Expand UNet input from 4 to 8 channels."""
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

    def _setup_optimizer(self) -> None:
        """Set up optimizer."""
        # Separate learning rates for new layers vs existing
        new_conv_params = list(self.unet.conv_in.parameters())
        controlnet_params = list(self.controlnet.parameters())
        unet_other_params = [
            p for n, p in self.unet.named_parameters() if "conv_in" not in n
        ]

        param_groups = [
            {"params": unet_other_params, "lr": self.config.learning_rate * 0.1},
            {"params": new_conv_params + controlnet_params, "lr": self.config.learning_rate},
        ]

        try:
            from bitsandbytes.optim import AdamW8bit
            self.optimizer = AdamW8bit(param_groups, weight_decay=self.config.weight_decay)
            logger.info("Using 8-bit AdamW optimizer")
        except ImportError:
            self.optimizer = torch.optim.AdamW(param_groups, weight_decay=self.config.weight_decay)
            logger.info("Using standard AdamW optimizer")

    def _setup_loss(self) -> None:
        """Set up loss function."""
        self.loss_fn = DiffusionLoss(loss_type="mse", reduction="mean")

    def _encode_images(
        self,
        images: Float[torch.Tensor, "B 3 H W"],
    ) -> Float[torch.Tensor, "B 4 h w"]:
        """Encode images to latent space."""
        # Normalize to [-1, 1]
        images = normalize_for_diffusion(images)

        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        return latents

    def _encode_text(
        self,
        input_ids: Int[torch.Tensor, "B L"],
    ) -> Float[torch.Tensor, "B L D"]:
        """Encode text to embeddings."""
        with torch.no_grad():
            text_embeddings = self.text_encoder(input_ids)[0]
        return text_embeddings

    def _get_empty_text_embeddings(
        self,
        batch_size: int,
    ) -> Float[torch.Tensor, "B L D"]:
        """Get embeddings for empty text (for conditioning dropout)."""
        empty_ids = self.tokenizer(
            "",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            empty_embeddings = self.text_encoder(empty_ids)[0]

        # Expand to batch size
        return empty_embeddings.expand(batch_size, -1, -1)

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> Float[torch.Tensor, ""]:
        """
        Perform a single training step.

        Args:
            batch: Batch from dataloader.

        Returns:
            Loss value.
        """
        # Move to device
        original = batch["original_image"].to(self.device)
        sketch = batch["sketch_image"].to(self.device)
        target = batch["target_image"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)

        batch_size = original.shape[0]

        # Encode images to latents
        target_latents = self._encode_images(target)
        original_latents = self._encode_images(original)

        # Add noise to target latents
        noise = torch.randn_like(target_latents)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)

        # Encode text
        text_embeddings = self._encode_text(input_ids)

        # Conditioning dropout
        if torch.rand(1).item() < self.config.conditioning_dropout_prob:
            text_embeddings = self._get_empty_text_embeddings(batch_size)

        # Concatenate noisy latents with original latents (8 channels total)
        concat_latents = torch.cat([noisy_latents, original_latents], dim=1)

        # Prepare sketch for ControlNet (normalize to [-1, 1])
        sketch_cond = normalize_for_diffusion(sketch)

        # Forward pass
        context_manager = autocast() if self.config.mixed_precision else torch.enable_grad()

        with context_manager:
            # ControlNet
            cn_output = self.controlnet(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=sketch_cond,
            )

            # UNet
            noise_pred = self.unet(
                sample=concat_latents,
                timestep=timesteps,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=cn_output.down_block_res_samples,
                mid_block_additional_residual=cn_output.mid_block_res_sample,
            ).sample

            # Loss
            loss = self.loss_fn(noise_pred, noise)

        return loss

    def train(self) -> None:
        """Run training loop."""
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.unet.train()
        self.controlnet.train()

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            for batch_idx, batch in enumerate(pbar):
                loss = self.train_step(batch)
                loss = loss / self.config.gradient_accumulation_steps

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            list(self.unet.parameters()) + list(self.controlnet.parameters()),
                            self.config.max_grad_norm,
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            list(self.unet.parameters()) + list(self.controlnet.parameters()),
                            self.config.max_grad_norm,
                        )
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    self.global_step += 1

                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                num_batches += 1

                # Logging
                if self.global_step % self.config.log_every_n_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "step": self.global_step})

                # Save checkpoint
                if self.global_step % self.config.save_every_n_steps == 0:
                    self.save_checkpoint(output_dir / f"checkpoint-{self.global_step}")

                # Early stopping based on max_steps
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    logger.info(f"Reached max_steps ({self.config.max_steps})")
                    break

            logger.info(f"Epoch {epoch} complete. Average loss: {epoch_loss / num_batches:.4f}")

            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

        # Save final checkpoint
        self.save_checkpoint(output_dir / "final")
        logger.info("Training complete")

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        path.mkdir(parents=True, exist_ok=True)

        self.unet.save_pretrained(path / "unet")
        self.controlnet.save_pretrained(path / "controlnet")

        # Save optimizer state
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "epoch": self.epoch,
            },
            path / "training_state.pt",
        )

        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        from diffusers import ControlNetModel, UNet2DConditionModel

        self.unet = UNet2DConditionModel.from_pretrained(path / "unet").to(self.device)
        self.controlnet = ControlNetModel.from_pretrained(path / "controlnet").to(self.device)

        training_state = torch.load(path / "training_state.pt")
        self.optimizer.load_state_dict(training_state["optimizer"])
        self.global_step = training_state["global_step"]
        self.epoch = training_state["epoch"]

        logger.info(f"Checkpoint loaded from {path}")
