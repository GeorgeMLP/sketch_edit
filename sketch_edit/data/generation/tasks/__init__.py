"""Editing-task implementations and registry."""

from sketch_edit.data.generation.tasks.base import EditingTask
from sketch_edit.data.generation.tasks.object_addition import ObjectAdditionTask
from sketch_edit.data.generation.tasks.object_replacement import ObjectReplacementTask
from sketch_edit.data.generation.tasks.orientation_change import OrientationChangeTask

TASK_REGISTRY: dict[str, type[EditingTask]] = {
    "object_addition": ObjectAdditionTask,
    "orientation_change": OrientationChangeTask,
    "object_replacement": ObjectReplacementTask,
}


def get_task(name: str) -> EditingTask:
    """Instantiate a registered editing task by *name*."""
    if name not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY))
        raise ValueError(f"Unknown task '{name}'. Available: {available}")
    return TASK_REGISTRY[name]()


def list_tasks() -> list[str]:
    """Return sorted list of registered task names."""
    return sorted(TASK_REGISTRY)


__all__ = [
    "EditingTask",
    "ObjectAdditionTask",
    "OrientationChangeTask",
    "ObjectReplacementTask",
    "TASK_REGISTRY",
    "get_task",
    "list_tasks",
]
