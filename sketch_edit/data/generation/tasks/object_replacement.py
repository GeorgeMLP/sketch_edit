"""Object replacement task — swap in a foreign object; model restores the original."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from sketch_edit.data.generation.tasks.base import EditingTask
from sketch_edit.utils import get_logger

logger = get_logger(__name__)


class ObjectReplacementTask(EditingTask):
    """Paste a foreign object over the original; the model must replace it back.

    * **Input** : image with a foreign object from a different class.
    * **Target**: original photo with the correct object.
    * **Sketch**: sketch of the correct object.
    """

    def __init__(self, seed: int = 42):
        self._seed = seed

    @property
    def name(self) -> str:
        return "object_replacement"

    @property
    def description(self) -> str:
        return "Replace a foreign object with the correct one using sketch guidance"

    # ------------------------------------------------------------------

    def generate_inputs(self, data_dir, segmentation_dir, output_dir, inpainter,
                        sample_ids, seg_results, skip_existing=True):
        assert inpainter is not None, "ObjectReplacementTask requires an inpainter"

        sampled_dir = Path(data_dir) / "sampled"
        mask_dir = Path(segmentation_dir) / "masks"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build class -> sample_ids mapping, then pair each sample with a
        # random sample from a *different* class.
        class_to_sids: dict[str, list[str]] = defaultdict(list)
        for sid, info in seg_results.items():
            class_to_sids[info["class"]].append(sid)

        rng = random.Random(self._seed)
        all_classes = sorted(class_to_sids)

        pairings: dict[str, str] = {}
        for sid in sample_ids:
            if sid not in seg_results:
                continue
            target_cls = seg_results[sid]["class"]
            other = [c for c in all_classes if c != target_cls]
            if not other:
                continue
            repl_cls = rng.choice(other)
            pairings[sid] = rng.choice(class_to_sids[repl_cls])

        results: dict[str, dict] = {}
        for sid in tqdm(sample_ids, desc="object_replacement"):
            input_path = output_dir / sid / "input.jpg"

            if skip_existing and input_path.exists() and sid in pairings:
                repl_sid = pairings[sid]
                results[sid] = self._meta(sid, repl_sid, input_path, seg_results)
                continue

            if sid not in pairings:
                continue
            repl_sid = pairings[sid]

            # Load target and replacement images/masks
            photo_files = list((sampled_dir / sid).glob("*_photo.jpg"))
            mask_path = mask_dir / f"{sid}_mask.png"
            repl_photos = list((sampled_dir / repl_sid).glob("*_photo.jpg"))
            repl_mask_path = mask_dir / f"{repl_sid}_mask.png"

            if not (photo_files and mask_path.exists() and repl_photos and repl_mask_path.exists()):
                continue

            try:
                image = Image.open(photo_files[0]).convert("RGB")
                mask = Image.open(mask_path).convert("L")
                repl_image = Image.open(repl_photos[0]).convert("RGB")
                repl_mask = Image.open(repl_mask_path).convert("L")

                composite = self._replace_object(
                    image, mask, seg_results[sid]["bbox"],
                    repl_image, repl_mask, seg_results[repl_sid]["bbox"],
                    inpainter,
                )

                input_path.parent.mkdir(parents=True, exist_ok=True)
                composite.save(input_path)

                results[sid] = self._meta(sid, repl_sid, input_path, seg_results)
            except Exception as e:
                logger.error(f"[object_replacement] sample {sid}: {e}")

        return results

    # ------------------------------------------------------------------

    @staticmethod
    def _replace_object(
        target_img: Image.Image,
        target_mask: Image.Image,
        target_bbox: list[int],
        repl_img: Image.Image,
        repl_mask: Image.Image,
        repl_bbox: list[int],
        inpainter,
    ) -> Image.Image:
        """Paste *repl* object over *target* object region."""
        background = inpainter.inpaint(target_img, target_mask)
        bg_np = np.array(background)

        # Crop replacement object
        repl_np = np.array(repl_img)
        repl_m = np.array(repl_mask)
        rx1, ry1, rx2, ry2 = repl_bbox
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2 = min(repl_np.shape[1], rx2)
        ry2 = min(repl_np.shape[0], ry2)
        crop_obj = repl_np[ry1:ry2, rx1:rx2]
        crop_mask = repl_m[ry1:ry2, rx1:rx2]
        if crop_obj.size == 0:
            raise ValueError("Empty crop for replacement object")

        # Resize to fit target bbox
        tx1, ty1, tx2, ty2 = target_bbox
        tx1, ty1 = max(0, tx1), max(0, ty1)
        tx2 = min(bg_np.shape[1], tx2)
        ty2 = min(bg_np.shape[0], ty2)
        tw, th = tx2 - tx1, ty2 - ty1
        if tw <= 0 or th <= 0:
            raise ValueError("Invalid target bounding box")

        resized_obj = cv2.resize(crop_obj, (tw, th), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(crop_mask, (tw, th), interpolation=cv2.INTER_NEAREST)

        result = bg_np.copy()
        region = result[ty1:ty2, tx1:tx2]
        region[resized_mask > 128] = resized_obj[resized_mask > 128]
        result[ty1:ty2, tx1:tx2] = region
        return Image.fromarray(result)

    @staticmethod
    def _meta(sid, repl_sid, input_path, seg_results):
        target_cls = seg_results[sid]["class"]
        repl_cls = seg_results[repl_sid]["class"]
        return {
            "input_path": str(input_path),
            "instruction": f"Replace the {repl_cls} with a {target_cls}.",
            "replacement_class": repl_cls,
            "replacement_sample_id": repl_sid,
        }
