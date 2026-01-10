"""Evaluation metrics for image editing quality."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from jaxtyping import UInt8
from PIL import Image

from sketch_edit.utils import get_logger

logger = get_logger(__name__)


def compute_mse(
    pred: UInt8[np.ndarray, "H W C"] | Image.Image,
    target: UInt8[np.ndarray, "H W C"] | Image.Image,
) -> float:
    """
    Compute Mean Squared Error between two images.

    Args:
        pred: Predicted image.
        target: Ground truth image.

    Returns:
        MSE value.
    """
    if isinstance(pred, Image.Image):
        pred = np.array(pred.convert("RGB"))
    if isinstance(target, Image.Image):
        target = np.array(target.convert("RGB"))

    # Resize if shapes don't match
    if pred.shape != target.shape:
        target = cv2.resize(target, (pred.shape[1], pred.shape[0]))

    pred_float = pred.astype(np.float32)
    target_float = target.astype(np.float32)

    # Use einops-style reduction
    diff_squared = (pred_float - target_float) ** 2
    mse = float(np.mean(diff_squared))

    return mse


def compute_histogram_intersection(
    pred: UInt8[np.ndarray, "H W C"] | Image.Image,
    target: UInt8[np.ndarray, "H W C"] | Image.Image,
    bins: int = 256,
) -> float:
    """
    Compute histogram intersection between two images.

    Higher values indicate more similar color distributions.

    Args:
        pred: Predicted image.
        target: Ground truth image.
        bins: Number of histogram bins.

    Returns:
        Average histogram intersection across channels (0 to 1).
    """
    if isinstance(pred, Image.Image):
        pred = np.array(pred.convert("RGB"))
    if isinstance(target, Image.Image):
        target = np.array(target.convert("RGB"))

    # Resize if shapes don't match
    if pred.shape != target.shape:
        target = cv2.resize(target, (pred.shape[1], pred.shape[0]))

    intersections = []
    for ch in range(3):
        h1 = cv2.calcHist([pred], [ch], None, [bins], [0, 256])
        h2 = cv2.calcHist([target], [ch], None, [bins], [0, 256])

        # Normalize histograms
        h1 = h1 / h1.sum()
        h2 = h2 / h2.sum()

        # Compute intersection
        intersection = float(np.minimum(h1, h2).sum())
        intersections.append(intersection)

    return float(np.mean(intersections))


def compute_clip_similarity(
    pred: UInt8[np.ndarray, "H W C"] | Image.Image,
    target: UInt8[np.ndarray, "H W C"] | Image.Image,
    model: torch.nn.Module | None = None,
    device: str | torch.device = "cuda",
) -> float:
    """
    Compute CLIP image embedding cosine similarity.

    Args:
        pred: Predicted image.
        target: Ground truth image.
        model: Pre-loaded CLIP model. If None, loads automatically.
        device: Device for model inference.

    Returns:
        Cosine similarity (0 to 1, higher is better).
    """
    from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
    from transformers import CLIPModel

    if isinstance(pred, np.ndarray):
        pred = Image.fromarray(pred)
    if isinstance(target, np.ndarray):
        target = Image.fromarray(target)

    # Load model if not provided
    if model is None:
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        model = model.to(device)
        model.eval()

    # Image preprocessing
    preprocess = Compose([
        Resize(224, interpolation=Image.Resampling.BICUBIC),
        CenterCrop((224, 224)),
        lambda img: img.convert("RGB"),
        ToTensor(),
        Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    pred_tensor = preprocess(pred).unsqueeze(0).to(device)
    target_tensor = preprocess(target).unsqueeze(0).to(device)

    with torch.no_grad():
        # Get image embeddings
        pred_features = model.vision_model(pixel_values=pred_tensor)[1]
        pred_features = model.visual_projection(pred_features)
        pred_features = pred_features / pred_features.norm(dim=-1, keepdim=True)

        target_features = model.vision_model(pixel_values=target_tensor)[1]
        target_features = model.visual_projection(target_features)
        target_features = target_features / target_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(pred_features, target_features)

    return float(similarity.item())


def compute_lpips(
    pred: UInt8[np.ndarray, "H W C"] | Image.Image,
    target: UInt8[np.ndarray, "H W C"] | Image.Image,
    loss_fn: torch.nn.Module | None = None,
    device: str | torch.device = "cuda",
) -> float:
    """
    Compute LPIPS perceptual distance.

    Lower values indicate more similar images perceptually.

    Args:
        pred: Predicted image.
        target: Ground truth image.
        loss_fn: Pre-loaded LPIPS loss function.
        device: Device for model inference.

    Returns:
        LPIPS distance (0 to 1, lower is better).
    """
    try:
        import lpips
    except ImportError:
        raise ImportError("lpips not installed. Run: pip install lpips")

    if isinstance(pred, np.ndarray):
        pred = Image.fromarray(pred)
    if isinstance(target, np.ndarray):
        target = Image.fromarray(target)

    # Load model if not provided
    if loss_fn is None:
        loss_fn = lpips.LPIPS(net="vgg").to(device)
        loss_fn.eval()

    # Convert to tensors in [-1, 1]
    from torchvision.transforms import Resize, ToTensor

    resize = Resize((256, 256))
    to_tensor = ToTensor()

    pred_tensor = to_tensor(resize(pred.convert("RGB"))).unsqueeze(0).to(device)
    target_tensor = to_tensor(resize(target.convert("RGB"))).unsqueeze(0).to(device)

    # Normalize to [-1, 1]
    pred_tensor = pred_tensor * 2 - 1
    target_tensor = target_tensor * 2 - 1

    with torch.no_grad():
        distance = loss_fn(pred_tensor, target_tensor)

    return float(distance.item())


def compute_all_metrics(
    pred: UInt8[np.ndarray, "H W C"] | Image.Image,
    target: UInt8[np.ndarray, "H W C"] | Image.Image,
    include_clip: bool = True,
    include_lpips: bool = False,
    clip_model: torch.nn.Module | None = None,
    lpips_fn: torch.nn.Module | None = None,
    device: str | torch.device = "cuda",
) -> dict[str, float]:
    """
    Compute all available metrics.

    Args:
        pred: Predicted image.
        target: Ground truth image.
        include_clip: Whether to compute CLIP similarity.
        include_lpips: Whether to compute LPIPS distance.
        clip_model: Pre-loaded CLIP model.
        lpips_fn: Pre-loaded LPIPS function.
        device: Device for neural network metrics.

    Returns:
        Dictionary of metric names to values.
    """
    metrics = {
        "mse": compute_mse(pred, target),
        "hist_intersection": compute_histogram_intersection(pred, target),
    }

    if include_clip:
        try:
            metrics["clip_similarity"] = compute_clip_similarity(
                pred, target, model=clip_model, device=device
            )
        except Exception as e:
            logger.warning(f"Could not compute CLIP similarity: {e}")

    if include_lpips:
        try:
            metrics["lpips"] = compute_lpips(pred, target, loss_fn=lpips_fn, device=device)
        except Exception as e:
            logger.warning(f"Could not compute LPIPS: {e}")

    return metrics
