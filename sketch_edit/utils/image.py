"""Image processing utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float, UInt8
from PIL import Image, ImageOps


def load_image(path: str | Path, mode: str = "RGB") -> Image.Image:
    """
    Load an image from disk.

    Args:
        path: Path to image file.
        mode: PIL image mode (e.g., "RGB", "L", "RGBA").

    Returns:
        PIL Image in specified mode.
    """
    return Image.open(path).convert(mode)


def save_image(image: Image.Image, path: str | Path) -> None:
    """
    Save a PIL image to disk.

    Args:
        image: PIL Image to save.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def resize_image(
    image: Image.Image,
    size: tuple[int, int],
    resample: int = Image.Resampling.BILINEAR,
) -> Image.Image:
    """
    Resize an image to the specified size.

    Args:
        image: PIL Image to resize.
        size: Target size as (width, height).
        resample: Resampling filter.

    Returns:
        Resized PIL Image.
    """
    return image.resize(size, resample=resample)


def invert_image(image: Image.Image) -> Image.Image:
    """
    Invert image colors.

    Args:
        image: PIL Image to invert.

    Returns:
        Inverted PIL Image.
    """
    return ImageOps.invert(image)


def image_to_tensor(
    image: Image.Image,
    normalize: bool = True,
) -> Float[torch.Tensor, "C H W"]:
    """
    Convert PIL Image to tensor.

    Args:
        image: PIL Image (RGB or L mode).
        normalize: If True, normalize to [0, 1]. Otherwise, keep [0, 255].

    Returns:
        Tensor of shape (C, H, W).
    """
    arr = np.array(image)

    if arr.ndim == 2:
        # Grayscale: add channel dimension
        arr = arr[:, :, np.newaxis]

    # HWC -> CHW
    tensor = rearrange(torch.from_numpy(arr), "h w c -> c h w")

    if normalize:
        tensor = tensor.float() / 255.0
    else:
        tensor = tensor.float()

    return tensor


def tensor_to_image(
    tensor: Float[torch.Tensor, "C H W"],
    denormalize: bool = True,
) -> Image.Image:
    """
    Convert tensor to PIL Image.

    Args:
        tensor: Tensor of shape (C, H, W) in [0, 1] or [-1, 1].
        denormalize: If True, assumes input is in [0, 1] or [-1, 1] and converts to [0, 255].

    Returns:
        PIL Image.
    """
    tensor = tensor.detach().cpu()

    if denormalize:
        # Clamp to valid range
        tensor = tensor.clamp(0, 1)
        tensor = tensor * 255

    # CHW -> HWC
    arr: UInt8[np.ndarray, "H W C"] = rearrange(tensor, "c h w -> h w c").numpy().astype(np.uint8)

    if arr.shape[2] == 1:
        arr = arr.squeeze(2)
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr, mode="RGB")


def sketch_to_mask(
    sketch_path: str | Path,
    output_path: str | Path | None = None,
    thresh_value: int = 200,
    closing_kernel_size: int = 15,
) -> UInt8[np.ndarray, "H W"]:
    """
    Convert a sketch image to a filled binary mask.

    Uses morphological operations to close gaps and fill contours.

    Args:
        sketch_path: Path to input sketch image.
        output_path: Optional path to save the mask.
        thresh_value: Threshold for binarization (default: 200).
        closing_kernel_size: Kernel size for morphological closing.

    Returns:
        Binary mask as numpy array (H, W) with values 0 or 255.
    """
    # Load and convert to grayscale
    img = cv2.imread(str(sketch_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {sketch_path}")

    # Invert + threshold: lines become white on black
    _, thresh = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY_INV)

    # Morphological closing to bridge gaps
    kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find external contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw filled contours into mask
    mask = np.zeros_like(closed)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), mask)

    return mask
