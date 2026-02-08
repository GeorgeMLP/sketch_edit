"""Orientation-change task — flip an object; model restores correct orientation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from sketch_edit.data.generation.tasks.base import EditingTask
from sketch_edit.utils import get_logger

logger = get_logger(__name__)


class OrientationChangeTask(EditingTask):
    """Flip the object horizontally; the model must flip it back.

    * **Input** : image with object horizontally flipped.
    * **Target**: original photo (correct orientation).
    * **Sketch**: sketch of the object in correct orientation.
    """

    @property
    def name(self) -> str:
        return "orientation_change"

    @property
    def description(self) -> str:
        return "Correct the orientation of a horizontally flipped object"

    # ------------------------------------------------------------------

    def generate_inputs(self, data_dir, segmentation_dir, output_dir, inpainter,
                        sample_ids, seg_results, skip_existing=True):
        assert inpainter is not None, "OrientationChangeTask requires an inpainter"

        sampled_dir = Path(data_dir) / "sampled"
        mask_dir = Path(segmentation_dir) / "masks"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, dict] = {}
        for sid in tqdm(sample_ids, desc="orientation_change"):
            input_path = output_dir / sid / "input.jpg"

            if skip_existing and input_path.exists() and sid in seg_results:
                results[sid] = {
                    "input_path": str(input_path),
                    "instruction": self._instruction(seg_results[sid]["class"]),
                }
                continue

            photo_files = list((sampled_dir / sid).glob("*_photo.jpg"))
            mask_path = mask_dir / f"{sid}_mask.png"
            if not photo_files or not mask_path.exists() or sid not in seg_results:
                continue

            try:
                image = Image.open(photo_files[0]).convert("RGB")
                mask = Image.open(mask_path).convert("L")
                composite = self._flip_object(image, mask, inpainter)

                input_path.parent.mkdir(parents=True, exist_ok=True)
                composite.save(input_path)

                results[sid] = {
                    "input_path": str(input_path),
                    "instruction": self._instruction(seg_results[sid]["class"]),
                }
            except Exception as e:
                logger.error(f"[orientation_change] sample {sid}: {e}")

        return results

    # ------------------------------------------------------------------

    @staticmethod
    def _flip_object(image: Image.Image, mask: Image.Image, inpainter) -> Image.Image:
        """Return *image* with the masked object horizontally flipped."""
        background = inpainter.inpaint(image, mask)

        img_np = np.array(image)
        mask_np = np.array(mask)
        bg_np = np.array(background)

        flipped_img = np.fliplr(img_np).copy()
        flipped_mask = np.fliplr(mask_np).copy()

        result = bg_np.copy()
        result[flipped_mask > 128] = flipped_img[flipped_mask > 128]
        return Image.fromarray(result)

    @staticmethod
    def _instruction(class_name: str) -> str:
        return f"Flip the {class_name} horizontally to correct its orientation."
