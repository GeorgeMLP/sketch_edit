"""Create a multi-task, split-aware dataset manifest."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sketch_edit.utils import get_logger, load_json, load_text_lines, save_json

logger = get_logger(__name__)


def assign_splits(
    sample_ids: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Deterministically assign sample IDs to train / val / test splits."""
    rng = np.random.RandomState(seed)
    ids = sorted(sample_ids)
    indices = np.arange(len(ids))
    rng.shuffle(indices)

    n = len(ids)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for i, idx in enumerate(indices):
        if i < n_train:
            splits["train"].append(ids[idx])
        elif i < n_train + n_val:
            splits["val"].append(ids[idx])
        else:
            splits["test"].append(ids[idx])

    for name, sids in splits.items():
        logger.info(f"  {name}: {len(sids)} samples")
    return splits


def create_dataset_manifest(
    data_dir: str | Path,
    output_file: str | Path,
    task_names: list[str] | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    validate: bool = True,
) -> dict:
    """Build a multi-task, split-aware manifest and write it to *output_file*.

    Directory layout assumed::

        data_dir/
            sampled/{id}/*_photo.jpg, *_sketch.png
            segmentation/segmentation_results.json
            generated/{task}/{id}/input.jpg
            generated/{task}/metadata.json
            instructions/{task}/{id}.txt      (optional VLM instructions)

    Returns the manifest dict ``{"metadata": …, "train": […], "val": […], "test": […]}``.
    """
    data_dir = Path(data_dir)
    sampled_dir = data_dir / "sampled"
    generated_dir = data_dir / "generated"
    seg_results_path = data_dir / "segmentation" / "segmentation_results.json"

    seg_results = load_json(seg_results_path) if seg_results_path.exists() else {}

    # Auto-discover tasks when not specified
    if task_names is None:
        task_names = (
            sorted(d.name for d in generated_dir.iterdir() if d.is_dir())
            if generated_dir.exists()
            else []
        )
    logger.info(f"Tasks: {task_names}")

    # All sample IDs
    sample_ids = sorted(d.name for d in sampled_dir.iterdir() if d.is_dir())

    # Assign splits and persist
    splits = assign_splits(sample_ids, train_ratio, val_ratio, seed)
    save_json(splits, data_dir / "splits.json")

    manifest: dict = {
        "metadata": {
            "num_samples": len(sample_ids),
            "tasks": task_names,
            "splits": {k: len(v) for k, v in splits.items()},
        },
        "train": [],
        "val": [],
        "test": [],
    }

    for task_name in task_names:
        task_meta_path = generated_dir / task_name / "metadata.json"
        task_meta = load_json(task_meta_path) if task_meta_path.exists() else {}

        for split_name, split_ids in splits.items():
            for sid in split_ids:
                if sid not in task_meta:
                    continue

                meta = task_meta[sid]
                input_path = meta["input_path"]
                if validate and not Path(input_path).exists():
                    continue

                sample_dir = sampled_dir / sid
                photo_files = list(sample_dir.glob("*_photo.jpg"))
                sketch_files = sorted(sample_dir.glob("*_sketch.png"))
                if not photo_files or not sketch_files:
                    continue

                # Prefer VLM instruction when available
                instruction = meta["instruction"]
                vlm_path = data_dir / "instructions" / task_name / f"{sid}.txt"
                if vlm_path.exists():
                    try:
                        lines = load_text_lines(vlm_path)
                        if lines:
                            instruction = lines[0]
                    except Exception:
                        pass

                class_name = seg_results.get(sid, {}).get("class", "")

                for sk_idx, sketch_path in enumerate(sketch_files):
                    manifest[split_name].append({
                        "id": f"{sid}_{task_name}_{sk_idx}",
                        "sample_id": sid,
                        "task": task_name,
                        "class_name": class_name,
                        "input_image_path": str(Path(input_path).absolute()),
                        "sketch_image_path": str(sketch_path.absolute()),
                        "target_image_path": str(photo_files[0].absolute()),
                        "edit_text": instruction,
                    })

    total = sum(len(manifest[s]) for s in ("train", "val", "test"))
    logger.info(f"Manifest: {total} entries across {len(task_names)} tasks")
    for s in ("train", "val", "test"):
        logger.info(f"  {s}: {len(manifest[s])}")

    save_json(manifest, output_file)
    logger.info(f"Manifest saved to {output_file}")
    return manifest
