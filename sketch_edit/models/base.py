"""Base model class and model registry for sketch-guided editing models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from sketch_edit.utils import get_logger

logger = get_logger(__name__)


class BaseEditModel(ABC):
    """
    Abstract base class for sketch+text-guided image editing models.

    Subclasses must implement:
    - load(): Load model weights
    - edit(): Perform image editing
    - get_trainable_params(): Return trainable parameters for fine-tuning
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the model.

        Args:
            device: Device to run the model on.
            dtype: Model dtype.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._is_loaded = False

    @abstractmethod
    def load(self, checkpoint_path: str | Path | None = None) -> None:
        """
        Load model weights.

        Args:
            checkpoint_path: Optional path to fine-tuned checkpoint.
        """
        pass

    @abstractmethod
    def edit(
        self,
        original_image: Image.Image,
        sketch_image: Image.Image,
        prompt: str,
        **kwargs: Any,
    ) -> Image.Image:
        """
        Edit an image based on sketch and text prompt.

        Args:
            original_image: Original image (background).
            sketch_image: Sketch of the object to add.
            prompt: Text instruction for the edit.
            **kwargs: Additional model-specific parameters.

        Returns:
            Edited image.
        """
        pass

    @abstractmethod
    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        """
        Get trainable parameters for fine-tuning.

        Returns:
            List of trainable parameters.
        """
        pass

    def save(self, output_path: str | Path) -> None:
        """
        Save model checkpoint.

        Args:
            output_path: Path to save the checkpoint.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        # Default implementation - subclasses can override
        logger.info(f"Saving model to {output_path}")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def to(self, device: str | torch.device) -> "BaseEditModel":
        """
        Move model to device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = device
        return self

    def enable_memory_optimizations(self) -> None:
        """Enable memory optimizations (xformers, gradient checkpointing, etc.)."""
        pass


class ModelRegistry:
    """Registry for available editing models."""

    _models: dict[str, type[BaseEditModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: type[BaseEditModel]) -> None:
        """
        Register a model class.

        Args:
            name: Model name for lookup.
            model_class: Model class to register.
        """
        cls._models[name] = model_class

    @classmethod
    def get(cls, name: str) -> type[BaseEditModel]:
        """
        Get a registered model class.

        Args:
            name: Model name.

        Returns:
            Model class.

        Raises:
            KeyError: If model name not found.
        """
        if name not in cls._models:
            available = list(cls._models.keys())
            raise KeyError(f"Model '{name}' not found. Available: {available}")
        return cls._models[name]

    @classmethod
    def list_models(cls) -> list[str]:
        """
        List all registered model names.

        Returns:
            List of model names.
        """
        return list(cls._models.keys())

    @classmethod
    def create(
        cls,
        name: str,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float16,
        **kwargs: Any,
    ) -> BaseEditModel:
        """
        Create a model instance.

        Args:
            name: Model name.
            device: Device to run on.
            dtype: Model dtype.
            **kwargs: Additional model-specific arguments.

        Returns:
            Model instance.
        """
        model_class = cls.get(name)
        return model_class(device=device, dtype=dtype, **kwargs)
