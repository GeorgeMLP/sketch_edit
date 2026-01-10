"""Model implementations for sketch-guided image editing."""

from sketch_edit.models.base import BaseEditModel, ModelRegistry
from sketch_edit.models.controlnet_scribble import ControlNetScribbleModel
from sketch_edit.models.controlnet_inpaint import ControlNetInpaintModel
from sketch_edit.models.instruct_pix2pix import InstructPix2PixModel

# Register all models
ModelRegistry.register("controlnet_scribble", ControlNetScribbleModel)
ModelRegistry.register("controlnet_inpaint", ControlNetInpaintModel)
ModelRegistry.register("instruct_pix2pix", InstructPix2PixModel)

__all__ = [
    "BaseEditModel",
    "ModelRegistry",
    "ControlNetScribbleModel",
    "ControlNetInpaintModel",
    "InstructPix2PixModel",
]
