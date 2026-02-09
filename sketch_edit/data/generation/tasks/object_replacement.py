"""Object replacement task — swap in a foreign object; model restores the original."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from sketch_edit.data.generation.insert_anything_utils import InsertAnythingCompositor
from sketch_edit.data.generation.tasks.base import EditingTask
from sketch_edit.utils import get_logger

logger = get_logger(__name__)


class ObjectReplacementTask(EditingTask):
    """Paste a foreign object over the original; the model must replace it back.

    Uses Insert-Anything (FLUX-based diffusion) for natural object composition
    with proper lighting, shadows, and perspective.

    * **Input** : image with a foreign object from a different class.
    * **Target**: original photo with the correct object.
    * **Sketch**: sketch of the correct object.
    """

    def __init__(
        self, 
        seed: int = 42, 
        compositor_seed: int = 666,
        num_inference_steps: int = 10,
        enable_cpu_offload: bool = False,
        enable_vae_slicing: bool = False,
    ):
        """Initialize the object replacement task.
        
        Args:
            seed: Random seed for sample pairing.
            compositor_seed: Seed for Insert-Anything generation.
            num_inference_steps: Number of diffusion steps (10=fast, 50=slow/high-quality).
            enable_cpu_offload: Use CPU offloading to reduce VRAM (slower, ~10GB vs 24GB).
            enable_vae_slicing: Enable VAE slicing to reduce memory spikes.
        """
        self._seed = seed
        self._compositor_seed = compositor_seed
        self._num_inference_steps = num_inference_steps
        self._enable_cpu_offload = enable_cpu_offload
        self._enable_vae_slicing = enable_vae_slicing

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

        # Initialize Insert-Anything compositor (shared across all samples)
        lora_path = Path("pretrained_models/20250321_steps5000_pytorch_lora_weights.safetensors")
        if not lora_path.exists():
            raise FileNotFoundError(
                f"Insert-Anything LoRA not found: {lora_path}\n"
                "Please download from https://huggingface.co/WensongSong/Insert-Anything"
            )
        
        logger.info("Initializing Insert-Anything compositor...")
        logger.info(f"  Inference steps: {self._num_inference_steps}")
        logger.info(f"  CPU offload: {self._enable_cpu_offload}")
        logger.info(f"  VAE slicing: {self._enable_vae_slicing}")
        
        compositor = InsertAnythingCompositor(
            lora_path=lora_path,
            device="cuda",
            enable_cpu_offload=self._enable_cpu_offload,
            enable_vae_slicing=self._enable_vae_slicing,
        )

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
                tar_image = np.array(Image.open(photo_files[0]).convert("RGB"))
                tar_mask = np.array(Image.open(mask_path).convert("L"))
                ref_image = np.array(Image.open(repl_photos[0]).convert("RGB"))
                ref_mask = np.array(Image.open(repl_mask_path).convert("L"))

                # Step 1: Remove original object from target image using inpainting
                tar_image_pil = Image.fromarray(tar_image)
                tar_mask_pil = Image.fromarray(tar_mask)
                clean_background = inpainter.inpaint(tar_image_pil, tar_mask_pil)
                clean_background_np = np.array(clean_background)

                # Step 2: Use Insert-Anything to naturally composite replacement onto clean background
                composite = compositor.composite(
                    ref_image=ref_image,
                    ref_mask=ref_mask,
                    tar_image=clean_background_np,
                    tar_mask=tar_mask,
                    seed=self._compositor_seed,
                    num_inference_steps=self._num_inference_steps,
                )

                input_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(composite).save(input_path)

                results[sid] = self._meta(sid, repl_sid, input_path, seg_results)
            except Exception as e:
                logger.error(f"[object_replacement] sample {sid}: {e}")

        return results

    # ------------------------------------------------------------------

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
