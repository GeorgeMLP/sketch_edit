"""Generate editing instructions using vision-language models."""

from __future__ import annotations

from pathlib import Path

import torch
from jaxtyping import Int
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from sketch_edit.utils import get_logger, save_text_lines, load_text_lines

logger = get_logger(__name__)


# Prompt for generating editing instructions
INSTRUCTION_PROMPT = (
    "We are building a dataset for image editing. Given a pair of images "
    "(a source image and a target image), please generate a detailed natural "
    "language instruction that describes how to edit the first image to "
    "transform it into the second image. Generate the instruction on one line "
    "and do not output anything other than the instruction.\n\n"
    "The source and target images are given as follows. Make sure your "
    "instructions accurately describe their differences.\n"
)

# Prompt for generating image descriptions
DESCRIPTION_PROMPT = (
    "Please generate a detailed description of the following image. Generate "
    "the description on one line and do not output anything other than the "
    "description.\n"
)


class InstructionGenerator:
    """
    Generate editing instructions using vision-language models.

    Uses Gemma-3 to generate natural language instructions describing
    how to edit a source image to match a target image.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-3-4b-it",
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the instruction generator.

        Args:
            model_id: HuggingFace model ID for Gemma-3.
            device: Device to run the model on.
            dtype: Model dtype (bfloat16 recommended).
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        logger.info(f"Loading {model_id} on {self.device}")
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            padding_side="left",
            use_fast=True,
        )
        logger.info("Model loaded successfully")

    @torch.no_grad()
    def generate_instruction(
        self,
        source_image: Image.Image,
        target_image: Image.Image,
        max_new_tokens: int = 500,
    ) -> str:
        """
        Generate an editing instruction for a source-target image pair.

        Args:
            source_image: Source image (before editing).
            target_image: Target image (after editing).
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated instruction string.
        """
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": INSTRUCTION_PROMPT},
                    {"type": "image", "image": source_image},
                    {"type": "image", "image": target_image},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.device)

        output: Int[torch.Tensor, "1 seq"] = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        response: str = self.processor.decode(output[0], skip_special_tokens=True)

        # Extract only the instruction (last non-empty line)
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        return lines[-1] if lines else ""

    @torch.no_grad()
    def generate_description(
        self,
        image: Image.Image,
        max_new_tokens: int = 500,
    ) -> str:
        """
        Generate a description for an image.

        Args:
            image: Image to describe.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated description string.
        """
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DESCRIPTION_PROMPT},
                    {"type": "image", "image": image},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.device)

        output: Int[torch.Tensor, "1 seq"] = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        response: str = self.processor.decode(output[0], skip_special_tokens=True)

        # Extract only the description (last non-empty line)
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        return lines[-1] if lines else ""

    def generate_instructions_batch(
        self,
        data_dir: str | Path,
        output_dir: str | Path,
        num_samples: int | None = None,
        skip_existing: bool = True,
    ) -> None:
        """
        Generate instructions for all samples in a dataset directory.

        Expects directory structure:
        - data_dir/sampled/{id}/*_photo.jpg (target images with objects)
        - data_dir/inpainted/{id}/*_remove.jpg (source images without objects)

        Args:
            data_dir: Root directory containing sampled/ and inpainted/.
            output_dir: Directory to save generated instructions.
            num_samples: Maximum number of samples to process (None for all).
            skip_existing: Skip samples that already have instructions.
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sampled_dir = data_dir / "sampled"
        inpainted_dir = data_dir / "inpainted"

        # Find all sample directories
        sample_ids = sorted([d.name for d in sampled_dir.iterdir() if d.is_dir()])
        if num_samples is not None:
            sample_ids = sample_ids[:num_samples]

        logger.info(f"Processing {len(sample_ids)} samples")

        for sample_id in tqdm(sample_ids, desc="Generating instructions"):
            output_path = output_dir / f"{sample_id}.txt"

            if skip_existing and output_path.exists():
                continue

            # Find images
            photo_files = list((sampled_dir / sample_id).glob("*_photo.jpg"))
            remove_files = list((inpainted_dir / sample_id).glob("*_remove.jpg"))

            if not photo_files or not remove_files:
                logger.warning(f"Missing images for sample {sample_id}")
                continue

            try:
                source_image = Image.open(remove_files[0]).convert("RGB")
                target_image = Image.open(photo_files[0]).convert("RGB")

                instruction = self.generate_instruction(source_image, target_image)
                save_text_lines([instruction], output_path)

            except Exception as e:
                logger.error(f"Failed to process sample {sample_id}: {e}")

        logger.info(f"Instructions saved to {output_dir}")


def load_instructions(path: str | Path) -> list[str]:
    """
    Load instructions from a text file.

    Args:
        path: Path to instruction file.

    Returns:
        List of instruction strings.
    """
    return load_text_lines(path)
