"""Base class for image editing tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sketch_edit.data.generation.inpainting import LamaInpainter


class EditingTask(ABC):
    """Base class for all image editing tasks.

    Each task defines how to generate input images from ground-truth samples.
    The original photos in ``sampled/`` are the targets; each task creates a
    modified version that the editing model must fix.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique task identifier (e.g. ``'object_addition'``)."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Short human-readable description."""
        ...

    @property
    def requires_inpainter(self) -> bool:
        """Whether this task needs a :class:`LamaInpainter`."""
        return True

    @abstractmethod
    def generate_inputs(
        self,
        data_dir: Path,
        segmentation_dir: Path,
        output_dir: Path,
        inpainter: LamaInpainter | None,
        sample_ids: list[str],
        seg_results: dict[str, dict],
        skip_existing: bool = True,
    ) -> dict[str, dict]:
        """Generate task-specific input images for the given samples.

        Args:
            data_dir: Root dataset directory containing ``sampled/``.
            segmentation_dir: Directory with ``masks/`` and results JSON.
            output_dir: Where to write generated inputs for **this** task.
            inpainter: Shared LaMa inpainter (``None`` when not required).
            sample_ids: Which samples to process.
            seg_results: ``{sample_id: {"class", "bbox", "center", …}}``.
            skip_existing: Skip samples whose output already exists.

        Returns:
            ``{sample_id: {"input_path": str, "instruction": str, …}}``
        """
        ...
