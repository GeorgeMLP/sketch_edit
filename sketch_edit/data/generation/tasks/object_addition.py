"""Object addition task — remove an object via inpainting; model adds it back."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from tqdm import tqdm

from sketch_edit.data.generation.tasks.base import EditingTask
from sketch_edit.utils import get_logger

logger = get_logger(__name__)


class ObjectAdditionTask(EditingTask):
    """Inpaint the object out; the model must add it back guided by sketch+text.

    * **Input** : image with object removed (LaMa inpainting).
    * **Target**: original photo with object present.
    * **Sketch**: sketch of the object to add.
    """

    @property
    def name(self) -> str:
        return "object_addition"

    @property
    def description(self) -> str:
        return "Add an object to the scene using sketch and text guidance"

    # ------------------------------------------------------------------

    def generate_inputs(self, data_dir, segmentation_dir, output_dir, inpainter,
                        sample_ids, seg_results, skip_existing=True):
        from sketch_edit.data.generation.inpainting import LamaInpainter  # type check

        assert inpainter is not None, "ObjectAdditionTask requires an inpainter"

        sampled_dir = Path(data_dir) / "sampled"
        mask_dir = Path(segmentation_dir) / "masks"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, dict] = {}
        for sid in tqdm(sample_ids, desc="object_addition"):
            input_path = output_dir / sid / "input.jpg"

            # fast-path: reuse existing
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
                inpainted = inpainter.inpaint(image, mask)

                input_path.parent.mkdir(parents=True, exist_ok=True)
                inpainted.save(input_path)

                results[sid] = {
                    "input_path": str(input_path),
                    "instruction": self._instruction(seg_results[sid]["class"]),
                }
            except Exception as e:
                logger.error(f"[object_addition] sample {sid}: {e}")

        return results

    @staticmethod
    def _instruction(class_name: str) -> str:
        return f"Add a {class_name} to the image."
