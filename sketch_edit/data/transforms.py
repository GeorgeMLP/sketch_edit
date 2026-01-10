"""Image transforms for dataset processing."""

from __future__ import annotations

import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torchvision import transforms


def default_transform(image_size: int = 512) -> transforms.Compose:
    """
    Create default image transform pipeline.

    Args:
        image_size: Target image size (square).

    Returns:
        Compose transform that resizes and converts to tensor.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),  # Converts to [0, 1]
    ])


def augment_transform(image_size: int = 512, flip_prob: float = 0.5) -> transforms.Compose:
    """
    Create augmentation transform pipeline for training.

    Args:
        image_size: Target image size (square).
        flip_prob: Probability of horizontal flip.

    Returns:
        Compose transform with augmentations.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=flip_prob),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
    ])


def normalize_for_diffusion(
    tensor: Float[torch.Tensor, "... C H W"],
) -> Float[torch.Tensor, "... C H W"]:
    """
    Normalize tensor from [0, 1] to [-1, 1] for diffusion models.

    Args:
        tensor: Input tensor in [0, 1] range.

    Returns:
        Tensor in [-1, 1] range.
    """
    return tensor * 2.0 - 1.0


def denormalize_from_diffusion(
    tensor: Float[torch.Tensor, "... C H W"],
) -> Float[torch.Tensor, "... C H W"]:
    """
    Denormalize tensor from [-1, 1] to [0, 1].

    Args:
        tensor: Input tensor in [-1, 1] range.

    Returns:
        Tensor in [0, 1] range.
    """
    return (tensor + 1.0) / 2.0


class PairedTransform:
    """
    Apply synchronized transforms to multiple images.

    Ensures the same random augmentation is applied to all images in a group.
    """

    def __init__(self, image_size: int = 512, augment: bool = False):
        """
        Initialize paired transform.

        Args:
            image_size: Target image size.
            augment: Whether to apply augmentation.
        """
        self.image_size = image_size
        self.augment = augment

    def __call__(
        self,
        original: Image.Image,
        sketch: Image.Image,
        target: Image.Image,
    ) -> tuple[
        Float[torch.Tensor, "3 H W"],
        Float[torch.Tensor, "3 H W"],
        Float[torch.Tensor, "3 H W"],
    ]:
        """
        Apply transforms to image triplet.

        Args:
            original: Original image (background without object).
            sketch: Sketch image.
            target: Target image (with object added).

        Returns:
            Tuple of transformed tensors (original, sketch, target).
        """
        # Resize all images
        resize = transforms.Resize(
            (self.image_size, self.image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        original = resize(original)
        sketch = resize(sketch)
        target = resize(target)

        # Apply synchronized augmentation
        if self.augment:
            # Random horizontal flip (same for all images)
            if torch.rand(1).item() > 0.5:
                original = transforms.functional.hflip(original)
                sketch = transforms.functional.hflip(sketch)
                target = transforms.functional.hflip(target)

        # Convert to tensors
        to_tensor = transforms.ToTensor()
        return to_tensor(original), to_tensor(sketch), to_tensor(target)
