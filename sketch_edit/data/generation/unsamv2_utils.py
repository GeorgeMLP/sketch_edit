"""Utilities for granularity-aware object segmentation using UnSAMv2.

UnSAMv2 extends SAM2 with the ability to segment objects at different granularities,
from fine parts (gra=0.1) to whole objects (gra=1.0).

Reference: https://github.com/yujunwei04/UnSAMv2
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Bool, UInt8
from PIL import Image

from sketch_edit.utils import get_logger

logger = get_logger(__name__)

# Add parent directory to path so sam2 package can be imported
_sketch_edit_path = Path(__file__).parent.parent.parent
if str(_sketch_edit_path) not in sys.path:
    sys.path.insert(0, str(_sketch_edit_path))


class UnSAMv2Segmenter:
    """Granularity-aware object segmentation using UnSAMv2.

    UnSAMv2 allows segmenting objects at different levels of detail:
    - gra=0.1: Fine parts (e.g., just the wing of a plane)
    - gra=0.5: Medium parts (e.g., fuselage + one wing)
    - gra=1.0: Whole object (similar to standard SAM)
    """

    def __init__(
        self,
        checkpoint_path: str | Path = "pretrained_models/unsamv2_plus_ckpt.pt",
        config_name: str = "configs/unsamv2_small",
        device: str | torch.device = "cuda",
    ):
        """Initialize UnSAMv2 model.

        Args:
            checkpoint_path: Path to UnSAMv2 or UnSAMv2+ checkpoint (.pt file).
                - UnSAMv2: 'pretrained_models/unsamv2_ckpt.pt'
                - UnSAMv2+: 'pretrained_models/unsamv2_plus_ckpt.pt' (better performance)
            config_name: Name of config file (uses same config for both variants).
            device: Device to run model on.
        """
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = torch.device(device)
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"UnSAMv2 checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading UnSAMv2 model from {checkpoint_path}")
        sam2_model = build_sam2(
            config_file=config_name,
            ckpt_path=str(checkpoint_path),
            device=str(self.device),
            mode="eval",
        )
        self.predictor = SAM2ImagePredictor(sam2_model)
        logger.info("UnSAMv2 model loaded successfully")

    @torch.no_grad()
    def segment_with_granularity(
        self,
        image: Image.Image | UInt8[np.ndarray, "H W 3"],
        bbox: list[int],
        granularity: float,
        multimask_output: bool = True,
    ) -> Bool[np.ndarray, "H W"]:
        """Segment object at specified granularity level.

        Args:
            image: Input image (PIL or numpy RGB array).
            bbox: Bounding box [x1, y1, x2, y2] around the object.
            granularity: Segmentation granularity (0.1=parts, 1.0=whole object).
            multimask_output: Whether to return multiple mask candidates.

        Returns:
            Binary mask (H, W) with 0/1 values for the segmented region.
        """
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image.convert("RGB"))
        else:
            image_np = image

        # Set image (computes embeddings)
        self.predictor.set_image(image_np)

        # Use bbox center as point prompt
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        point_coords = np.array([[center_x, center_y]])
        point_labels = np.array([1])  # foreground

        # Predict with granularity control
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output,
            gra=granularity,
        )

        # Select best mask based on IoU scores
        if multimask_output:
            best_idx = int(np.argmax(scores))
            mask = masks[best_idx]
            logger.debug(f"UnSAMv2 scores: {scores}, selected mask {best_idx} (gra={granularity})")
        else:
            mask = masks[0]

        # Ensure binary mask (0 or 1)
        return (mask > 0.5).astype(bool)
