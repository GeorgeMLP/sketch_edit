"""Object segmentation using Grounding DINO + SAM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from jaxtyping import Bool, Float
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)

from sketch_edit.utils import get_logger, save_json, load_json

logger = get_logger(__name__)


class GroundedSegmenter:
    """
    Object segmentation using Grounding DINO + SAM.

    Grounding DINO detects objects from text prompts, then SAM segments them.
    """

    def __init__(
        self,
        dino_model_id: str = "IDEA-Research/grounding-dino-tiny",
        sam_model_id: str = "facebook/sam-vit-huge",
        device: str | torch.device | None = None,
    ):
        """
        Initialize the segmenter.

        Args:
            dino_model_id: HuggingFace model ID for Grounding DINO.
            sam_model_id: HuggingFace model ID for SAM.
            device: Device to run models on.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading Grounding DINO from {dino_model_id}")
        self.dino_processor = AutoProcessor.from_pretrained(dino_model_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(
            self.device
        )

        logger.info(f"Loading SAM from {sam_model_id}")
        self.sam_processor = SamProcessor.from_pretrained(sam_model_id)
        self.sam_model = SamModel.from_pretrained(sam_model_id).to(self.device)

        logger.info("Models loaded successfully")

    @torch.no_grad()
    def segment(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
    ) -> dict[str, Float[np.ndarray, "..."] | list[int] | tuple[int, int]]:
        """
        Segment an object in an image based on a text prompt.

        Args:
            image: Input image.
            text_prompt: Text description of object to segment (e.g., "tiger").
            box_threshold: Confidence threshold for bounding box detection.
            text_threshold: Confidence threshold for text matching.

        Returns:
            Dictionary containing:
            - mask: Binary mask array (H, W) with values 0 or 255
            - bbox: Bounding box [x1, y1, x2, y2]
            - center: Center point (x, y) of the mask
        """
        image_rgb = image.convert("RGB")

        # Run Grounding DINO
        dino_inputs = self.dino_processor(
            images=image_rgb, text=[[text_prompt]], return_tensors="pt"
        ).to(self.device)

        dino_outputs = self.dino_model(**dino_inputs)
        dino_results = self.dino_processor.post_process_grounded_object_detection(
            dino_outputs,
            dino_inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image_rgb.size[::-1]],  # (H, W)
        )

        if len(dino_results[0]["boxes"]) == 0:
            raise ValueError(f"No object detected for prompt: {text_prompt}")

        # Get the top detection
        box: list[int] = [int(x) for x in dino_results[0]["boxes"][0].tolist()]
        x1, y1, x2, y2 = box

        # Run SAM with the detected bounding box
        sam_inputs = self.sam_processor(
            image_rgb, input_boxes=[[[x1, y1, x2, y2]]], return_tensors="pt"
        ).to(self.device)

        sam_outputs = self.sam_model(**sam_inputs)
        masks = self.sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu(),
        )[0]

        # Get the best mask
        mask: Bool[np.ndarray, "H W"] = masks[0][0].numpy()

        # Compute center from mask
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            raise ValueError("Empty mask produced")

        center_x = int(np.mean(xs))
        center_y = int(np.mean(ys))

        # Convert mask to uint8
        mask_uint8: Float[np.ndarray, "H W"] = (mask.astype(np.float32) * 255).astype(np.uint8)

        return {
            "mask": mask_uint8,
            "bbox": box,
            "center": (center_x, center_y),
        }

    def segment_dataset(
        self,
        data_dir: str | Path,
        output_dir: str | Path,
        results_file: str = "segmentation_results.json",
        skip_existing: bool = True,
    ) -> None:
        """
        Segment objects in all images in a dataset directory.

        Expects directory structure:
        - data_dir/sampled/{id}/*_photo.jpg

        The text prompt is extracted from the filename (e.g., "apple_n07739125_4340_photo.jpg" -> "apple").

        Args:
            data_dir: Root directory containing sampled/.
            output_dir: Directory to save masks and results.
            results_file: Name of JSON file to save results.
            skip_existing: Skip samples that already have masks.
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_dir = output_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

        sampled_dir = data_dir / "sampled"
        results_path = output_dir / results_file

        # Load existing results
        if results_path.exists():
            results = load_json(results_path)
        else:
            results = {}

        # Find all sample directories
        sample_ids = sorted([d.name for d in sampled_dir.iterdir() if d.is_dir()])
        logger.info(f"Processing {len(sample_ids)} samples")

        for sample_id in tqdm(sample_ids, desc="Segmenting objects"):
            mask_path = mask_dir / f"{sample_id}_mask.png"

            if skip_existing and mask_path.exists():
                continue

            # Find photo file
            photo_files = list((sampled_dir / sample_id).glob("*_photo.jpg"))
            if not photo_files:
                logger.warning(f"No photo found for sample {sample_id}")
                continue

            photo_path = photo_files[0]
            # Extract class name from filename
            class_name = photo_path.stem.split("_")[0]

            try:
                image = Image.open(photo_path).convert("RGB")
                result = self.segment(image, class_name)

                # Save mask
                mask_img = Image.fromarray(result["mask"]).convert("L")
                mask_img.save(mask_path)

                # Save result
                results[sample_id] = {
                    "class": class_name,
                    "bbox": result["bbox"],
                    "center": result["center"],
                    "mask_path": str(mask_path.relative_to(output_dir)),
                }

            except Exception as e:
                logger.error(f"Failed to segment sample {sample_id}: {e}")

        # Save results
        save_json(results, results_path)
        logger.info(f"Segmentation complete. Results saved to {results_path}")
