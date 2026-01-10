"""Loss functions for training diffusion models."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import reduce
from jaxtyping import Float


class DiffusionLoss:
    """
    Loss functions for diffusion model training.

    Supports various noise prediction objectives.
    """

    def __init__(
        self,
        loss_type: str = "mse",
        reduction: str = "mean",
    ):
        """
        Initialize loss function.

        Args:
            loss_type: Type of loss ("mse", "l1", "huber").
            reduction: Reduction method ("mean", "sum", "none").
        """
        self.loss_type = loss_type
        self.reduction = reduction

    def __call__(
        self,
        pred: Float[torch.Tensor, "B C H W"],
        target: Float[torch.Tensor, "B C H W"],
        weights: Float[torch.Tensor, "B"] | None = None,
    ) -> Float[torch.Tensor, ""]:
        """
        Compute loss between predicted and target noise.

        Args:
            pred: Predicted noise tensor.
            target: Target noise tensor.
            weights: Optional per-sample weights.

        Returns:
            Scalar loss value.
        """
        if self.loss_type == "mse":
            loss = F.mse_loss(pred.float(), target.float(), reduction="none")
        elif self.loss_type == "l1":
            loss = F.l1_loss(pred.float(), target.float(), reduction="none")
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(pred.float(), target.float(), reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Reduce over spatial and channel dimensions using einops
        # loss shape: (B, C, H, W) -> (B,)
        loss = reduce(loss, "b c h w -> b", reduction="mean")

        # Apply per-sample weights if provided
        if weights is not None:
            loss = loss * weights

        # Final reduction
        if self.reduction == "mean":
            return reduce(loss, "b -> ", reduction="mean")
        elif self.reduction == "sum":
            return reduce(loss, "b -> ", reduction="sum")
        else:
            return loss


class PerceptualLoss:
    """
    Perceptual loss using LPIPS for image quality.

    Optional loss for improving visual quality of generated images.
    """

    def __init__(
        self,
        net: str = "vgg",
        device: str | torch.device = "cuda",
    ):
        """
        Initialize perceptual loss.

        Args:
            net: Network to use ("vgg", "alex", "squeeze").
            device: Device to run on.
        """
        try:
            import lpips
            self.loss_fn = lpips.LPIPS(net=net).to(device)
            self.loss_fn.eval()
            for p in self.loss_fn.parameters():
                p.requires_grad_(False)
        except ImportError:
            raise ImportError("lpips not installed. Run: pip install lpips")

    def __call__(
        self,
        pred: Float[torch.Tensor, "B C H W"],
        target: Float[torch.Tensor, "B C H W"],
    ) -> Float[torch.Tensor, ""]:
        """
        Compute perceptual loss.

        Args:
            pred: Predicted images in [-1, 1].
            target: Target images in [-1, 1].

        Returns:
            Scalar loss value.
        """
        loss = self.loss_fn(pred, target)
        return reduce(loss, "b 1 1 1 -> ", reduction="mean")
