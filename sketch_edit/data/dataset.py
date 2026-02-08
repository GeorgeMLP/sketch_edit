"""Dataset classes for sketch-guided image editing (multi-task, split-aware)."""

from __future__ import annotations

from pathlib import Path

import torch
from jaxtyping import Float, Int
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

from sketch_edit.data.transforms import PairedTransform
from sketch_edit.utils import get_logger, load_json

logger = get_logger(__name__)


class SketchEditDataset(Dataset):
    """Multi-task dataset for sketch+text-guided image editing.

    Reads a split-aware manifest with the layout::

        {"metadata": …, "train": [entries…], "val": [entries…], "test": [entries…]}

    Each entry has ``input_image_path``, ``sketch_image_path``,
    ``target_image_path``, ``edit_text``, ``task``, etc.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        task: str | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        image_size: int = 512,
        augment: bool = False,
        max_text_length: int = 77,
    ):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.transform = PairedTransform(image_size=image_size, augment=augment)

        manifest = load_json(self.manifest_path)
        self.data: list[dict] = manifest[split]

        if task is not None:
            self.data = [e for e in self.data if e["task"] == task]

        logger.info(
            f"Loaded {len(self.data)} samples (split={split}"
            + (f", task={task}" if task else "")
            + f") from {self.manifest_path}"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx: int,
    ) -> dict[str, Float[torch.Tensor, "..."] | Int[torch.Tensor, "..."] | str]:
        item = self.data[idx]

        original = Image.open(item["input_image_path"]).convert("RGB")
        sketch = Image.open(item["sketch_image_path"]).convert("RGB")
        target = Image.open(item["target_image_path"]).convert("RGB")

        original_t, sketch_t, target_t = self.transform(original, sketch, target)

        result: dict = {
            "original_image": original_t,
            "sketch_image": sketch_t,
            "target_image": target_t,
            "edit_text": item["edit_text"],
            "task": item.get("task", ""),
            "sample_id": item.get("id", str(idx)),
        }

        if self.tokenizer is not None:
            tokens = self.tokenizer(
                item["edit_text"],
                max_length=self.max_text_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            result["input_ids"] = tokens.input_ids.squeeze(0)
            result["attention_mask"] = tokens.attention_mask.squeeze(0)

        return result


def create_dataloaders(
    manifest_path: str | Path,
    tokenizer: PreTrainedTokenizer | None = None,
    batch_size: int = 4,
    image_size: int = 512,
    task: str | None = None,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders from a split-aware manifest."""
    train_ds = SketchEditDataset(
        manifest_path, split="train", task=task, tokenizer=tokenizer,
        image_size=image_size, augment=True,
    )
    val_ds = SketchEditDataset(
        manifest_path, split="val", task=task, tokenizer=tokenizer,
        image_size=image_size, augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    logger.info(f"Dataloaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    return train_loader, val_loader
