"""Generate editing instructions using vision-language models."""

from __future__ import annotations

from pathlib import Path

import torch
from jaxtyping import Int
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from sketch_edit.utils import get_logger, load_json, load_text_lines, save_text_lines

logger = get_logger(__name__)

INSTRUCTION_PROMPT_TEMPLATE = (
    "We are building a dataset for image editing. Given a pair of images "
    "(a source image and a target image), please generate a detailed natural "
    "language instruction that describes how to edit the first image to "
    "transform it into the second image.\n\n"
    "Task context: {task_context}\n\n"
    "Given this context, please generate a specific instruction that describes "
    "how to edit the first image to match the second image. Generate the "
    "instruction on one line and do not output anything other than the instruction.\n\n"
    "The source and target images are given as follows:\n"
)


class InstructionGenerator:
    """Generate editing instructions using Gemma-3 (or similar VLM)."""

    def __init__(
        self,
        model_id: str = "google/gemma-3-4b-it",
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        logger.info(f"Loading {model_id} on {self.device}")
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.processor = AutoProcessor.from_pretrained(model_id, padding_side="left", use_fast=True)
        logger.info("Model loaded successfully")

    @torch.no_grad()
    def generate_instruction(
        self,
        source_image: Image.Image,
        target_image: Image.Image,
        task_context: str = "",
        max_new_tokens: int = 500,
    ) -> str:
        """Generate an editing instruction for a source→target image pair.
        
        Args:
            source_image: Input image (with edits to be made).
            target_image: Target image (desired result).
            task_context: Context about the editing task (e.g., "Replace the turtle with an apple").
            max_new_tokens: Maximum tokens to generate.
            
        Returns:
            Generated instruction string.
        """
        prompt = INSTRUCTION_PROMPT_TEMPLATE.format(task_context=task_context)
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": source_image},
                    {"type": "image", "image": target_image},
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.device)

        output: Int[torch.Tensor, "1 seq"] = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        response: str = self.processor.decode(output[0], skip_special_tokens=True)
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        return lines[-1] if lines else ""

    def generate_instructions_batch(
        self,
        data_dir: str | Path,
        output_dir: str | Path,
        task_name: str = "object_addition",
        num_samples: int | None = None,
        skip_existing: bool = True,
    ) -> None:
        """Generate VLM instructions for a specific editing task.

        Reads generated inputs from ``data_dir/generated/{task_name}/{id}/input.jpg``
        and original photos from ``data_dir/sampled/{id}/*_photo.jpg``.
        Also loads task metadata to provide context to the VLM about the editing operation.
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_dir = data_dir / "generated" / task_name
        sampled_dir = data_dir / "sampled"

        if not generated_dir.exists():
            logger.warning(f"No generated inputs for task '{task_name}'")
            return

        # Load task metadata to get editing context for each sample
        task_meta_path = generated_dir / "metadata.json"
        task_meta = load_json(task_meta_path) if task_meta_path.exists() else {}

        sample_ids = sorted([d.name for d in generated_dir.iterdir() if d.is_dir()])
        if num_samples is not None:
            sample_ids = sample_ids[:num_samples]

        logger.info(f"Generating VLM instructions for {len(sample_ids)} samples ({task_name})")

        for sid in tqdm(sample_ids, desc=f"Instructions ({task_name})"):
            out_path = output_dir / f"{sid}.txt"
            if skip_existing and out_path.exists():
                continue

            input_path = generated_dir / sid / "input.jpg"
            photo_files = list((sampled_dir / sid).glob("*_photo.jpg"))
            if not input_path.exists() or not photo_files:
                continue

            try:
                source = Image.open(input_path).convert("RGB")
                target = Image.open(photo_files[0]).convert("RGB")
                
                # Get task-specific context from metadata
                task_context = ""
                if sid in task_meta and "instruction" in task_meta[sid]:
                    task_context = task_meta[sid]["instruction"]
                
                instruction = self.generate_instruction(source, target, task_context=task_context)
                save_text_lines([instruction], out_path)
            except Exception as e:
                logger.error(f"Instruction generation failed for {sid}: {e}")

        logger.info(f"Instructions saved to {output_dir}")


def load_instructions(path: str | Path) -> list[str]:
    """Load instructions from a text file."""
    return load_text_lines(path)
