"""Dataset classes for sketch-guided image editing."""

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
    """
    Dataset for sketch+text-guided image editing.

    Each sample contains:
    - original_image: Background image without the target object
    - sketch_image: Sketch of the object to add
    - target_image: Ground truth image with the object
    - edit_text: Natural language instruction for the edit
    """

    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: PreTrainedTokenizer | None = None,
        image_size: int = 512,
        augment: bool = False,
        max_text_length: int = 77,
    ):
        """
        Initialize the dataset.

        Args:
            manifest_path: Path to JSON manifest file containing data paths.
            tokenizer: Text tokenizer for encoding instructions.
            image_size: Size to resize images to.
            augment: Whether to apply data augmentation.
            max_text_length: Maximum length for text tokenization.
        """
        self.manifest_path = Path(manifest_path)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.transform = PairedTransform(image_size=image_size, augment=augment)

        # Load manifest
        self.data = load_json(self.manifest_path)
        logger.info(f"Loaded {len(self.data)} samples from {self.manifest_path}")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Float[torch.Tensor, "..."] | Int[torch.Tensor, "..."] | str]:
        """
        Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing:
            - original_image: Tensor (3, H, W) in [0, 1]
            - sketch_image: Tensor (3, H, W) in [0, 1]
            - target_image: Tensor (3, H, W) in [0, 1]
            - input_ids: Token IDs (if tokenizer provided)
            - attention_mask: Attention mask (if tokenizer provided)
            - edit_text: Raw text instruction
            - sample_id: Unique sample identifier
        """
        item = self.data[idx]

        # Load images
        original = Image.open(item["original_image_path"]).convert("RGB")
        sketch = Image.open(item["sketch_image_path"]).convert("RGB")
        target = Image.open(item["target_image_path"]).convert("RGB")

        # Apply transforms
        original_tensor, sketch_tensor, target_tensor = self.transform(original, sketch, target)

        result: dict[str, Float[torch.Tensor, "..."] | Int[torch.Tensor, "..."] | str] = {
            "original_image": original_tensor,
            "sketch_image": sketch_tensor,
            "target_image": target_tensor,
            "edit_text": item["edit_text"],
            "sample_id": item.get("id", str(idx)),
        }

        # Tokenize text if tokenizer provided
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
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        manifest_path: Path to dataset manifest JSON.
        tokenizer: Text tokenizer.
        batch_size: Batch size for dataloaders.
        image_size: Image size for resizing.
        val_split: Fraction of data for validation.
        num_workers: Number of dataloader workers.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    import numpy as np

    # Create dataset
    dataset = SketchEditDataset(
        manifest_path=manifest_path,
        tokenizer=tokenizer,
        image_size=image_size,
        augment=True,
    )

    # Create validation dataset (no augmentation)
    val_dataset = SketchEditDataset(
        manifest_path=manifest_path,
        tokenizer=tokenizer,
        image_size=image_size,
        augment=False,
    )

    # Split indices
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    np.random.seed(seed)
    np.random.shuffle(indices)

    val_size = int(val_split * dataset_size)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    logger.info(f"Dataset split: {len(train_indices)} train, {len(val_indices)} val")

    # Create samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
