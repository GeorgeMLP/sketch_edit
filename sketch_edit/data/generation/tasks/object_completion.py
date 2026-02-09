"""Object completion task using UnSAMv2 granularity-aware segmentation.

This task segments and removes only part of an object (using random granularity between 0-1),
then asks the model to complete the missing region.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from sketch_edit.data.generation.tasks.base import EditingTask
from sketch_edit.data.generation.unsamv2_utils import UnSAMv2Segmenter
from sketch_edit.utils import get_logger

logger = get_logger(__name__)


class ObjectCompletionTask(EditingTask):
    """Complete a partially removed object.

    Uses UnSAMv2 to segment only part of an object (random granularity 0-1),
    removes the part, and generates instruction to complete the object.
    """

    @property
    def name(self) -> str:
        return "object_completion"

    @property
    def description(self) -> str:
        return "Complete a partially removed object using granularity-aware segmentation"

    def __init__(
        self,
        granularity_min: float = 0.2,
        granularity_max: float = 0.9,
        seed: int | None = None,
    ):
        """Initialize object completion task.

        Args:
            granularity_min: Minimum segmentation granularity (0.2 = small parts).
            granularity_max: Maximum segmentation granularity (0.9 = most of object).
            seed: Random seed for reproducibility.
        """
        super().__init__()
        self.granularity_min = granularity_min
        self.granularity_max = granularity_max
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_inputs(
        self,
        data_dir: Path,
        segmentation_dir: Path,
        output_dir: Path,
        inpainter,
        sample_ids: list[str],
        seg_results: dict[str, dict],
        skip_existing: bool = True,
    ) -> dict[str, dict]:
        """Generate partial object removal inputs for completion task.

        Args:
            data_dir: Root dataset directory containing ``sampled/``.
            segmentation_dir: Directory with ``masks/`` and results JSON.
            output_dir: Where to write generated inputs for **this** task.
            inpainter: Shared LaMa inpainter.
            sample_ids: Which samples to process.
            seg_results: ``{sample_id: {"class", "bbox", "center", ...}}``.
            skip_existing: Skip samples whose output already exists.

        Returns:
            ``{sample_id: {"input_path": str, "instruction": str, ...}}``
        """
        assert inpainter is not None, "ObjectCompletionTask requires an inpainter"

        sampled_dir = Path(data_dir) / "sampled"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize UnSAMv2 once
        logger.info("Initializing UnSAMv2 segmenter for object completion")
        unsamv2 = UnSAMv2Segmenter()

        results: dict[str, dict] = {}
        for sid in tqdm(sample_ids, desc="object_completion"):
            input_path = output_dir / sid / "input.jpg"

            # fast-path: reuse existing
            if skip_existing and input_path.exists() and sid in seg_results:
                results[sid] = {
                    "input_path": str(input_path),
                    "instruction": self._instruction(seg_results[sid]["class"]),
                }
                continue

            photo_files = list((sampled_dir / sid).glob("*_photo.jpg"))
            if not photo_files or sid not in seg_results:
                continue

            try:
                image = Image.open(photo_files[0]).convert("RGB")
                image_np = np.array(image)

                # Generate random granularity for partial segmentation
                granularity = random.uniform(self.granularity_min, self.granularity_max)
                seg_info = seg_results[sid]
                bbox = seg_info["bbox"]

                # Segment part of the object using UnSAMv2
                partial_mask = unsamv2.segment_with_granularity(
                    image=image,
                    bbox=bbox,
                    granularity=granularity,
                )

                # Inpaint the partial region
                partial_mask_pil = Image.fromarray((partial_mask * 255).astype(np.uint8))
                inpainted = inpainter.inpaint(image, partial_mask_pil)

                # Save results
                input_path.parent.mkdir(parents=True, exist_ok=True)
                inpainted.save(input_path)

                results[sid] = {
                    "input_path": str(input_path),
                    "instruction": self._instruction(seg_info["class"]),
                    "granularity": float(granularity),
                }
            except Exception as e:
                logger.error(f"[object_completion] sample {sid}: {e}")

        return results

    @staticmethod
    def _instruction(class_name: str) -> str:
        """Generate instruction for completing the object.

        Args:
            class_name: Name of the object class.

        Returns:
            Natural language instruction.
        """
        templates = [
            f"Complete the {class_name}",
            f"Fill in the missing part of the {class_name}",
            f"Restore the {class_name}",
            f"Complete the partial {class_name}",
        ]
        return random.choice(templates)
