"""Data generation modules for creating synthetic editing datasets."""

from sketch_edit.data.generation.inpainting import LamaInpainter
from sketch_edit.data.generation.instructions import InstructionGenerator
from sketch_edit.data.generation.manifest import create_dataset_manifest
from sketch_edit.data.generation.segmentation import GroundedSegmenter
from sketch_edit.data.generation.tasks import (
    TASK_REGISTRY,
    EditingTask,
    get_task,
    list_tasks,
)

__all__ = [
    "EditingTask",
    "GroundedSegmenter",
    "InstructionGenerator",
    "LamaInpainter",
    "TASK_REGISTRY",
    "create_dataset_manifest",
    "get_task",
    "list_tasks",
]
