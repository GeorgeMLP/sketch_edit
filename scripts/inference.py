"""Run inference with sketch+text-guided image editing models."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from sketch_edit.models import ModelRegistry
from sketch_edit.utils import get_logger, save_image, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with sketch+text-guided image editing models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model", type=str, default="controlnet_inpaint",
        choices=ModelRegistry.list_models(),
    )
    parser.add_argument("--checkpoint", type=str, default=None)

    # Single-image inference
    parser.add_argument("--original_image", type=str, default=None)
    parser.add_argument("--sketch", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--output", type=str, default="edited.png")

    # Batch inference
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument(
        "--split", type=str, default="test",
        help="Which manifest split to use for batch inference.",
    )
    parser.add_argument("--task", type=str, default=None, help="Filter to a specific editing task.")
    parser.add_argument("--output_dir", type=str, default="./predictions")
    parser.add_argument("--num_samples", type=int, default=None)

    # Generation
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--invert_sketch", action="store_true", default=True)
    parser.add_argument("--no_invert_sketch", action="store_false", dest="invert_sketch")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--log_file", type=str, default="logs/inference.log")

    return parser.parse_args()


def _torch_dtype(s: str):
    import torch
    return {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[s]


def run_single(args, logger):
    if not all([args.original_image, args.sketch, args.prompt]):
        logger.error("For single inference, provide --original_image, --sketch, and --prompt")
        return

    model = ModelRegistry.create(args.model, device=args.device, dtype=_torch_dtype(args.dtype))
    model.load(checkpoint_path=args.checkpoint)

    original = Image.open(args.original_image).convert("RGB")
    sketch = Image.open(args.sketch).convert("RGB")

    edited = model.edit(
        original_image=original, sketch_image=sketch, prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        invert_sketch=args.invert_sketch,
    )
    save_image(edited, args.output)
    logger.info(f"Saved → {args.output}")


def run_batch(args, logger):
    from tqdm import tqdm
    from sketch_edit.utils import load_json

    manifest = load_json(args.manifest)
    entries = manifest.get(args.split, manifest) if isinstance(manifest, dict) else manifest
    if isinstance(entries, dict):
        entries = entries.get(args.split, [])

    if args.task:
        entries = [e for e in entries if e.get("task") == args.task]
    if args.num_samples:
        entries = entries[: args.num_samples]

    logger.info(f"Processing {len(entries)} samples")

    model = ModelRegistry.create(args.model, device=args.device, dtype=_torch_dtype(args.dtype))
    model.load(checkpoint_path=args.checkpoint)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for entry in tqdm(entries, desc="Inference"):
        sid = entry["id"]
        out = output_dir / f"{sid}.png"
        if out.exists():
            continue
        try:
            original = Image.open(entry["input_image_path"]).convert("RGB")
            sketch = Image.open(entry["sketch_image_path"]).convert("RGB")
            edited = model.edit(
                original_image=original, sketch_image=sketch, prompt=entry["edit_text"],
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                image_guidance_scale=args.image_guidance_scale,
                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                invert_sketch=args.invert_sketch,
            )
            save_image(edited, out)
        except Exception as e:
            logger.error(f"Error on {sid}: {e}")

    logger.info(f"Predictions saved → {output_dir}")


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)
    logger = get_logger(__name__)

    logger.info("=" * 50)
    logger.info("Starting inference")

    if args.manifest:
        run_batch(args, logger)
    else:
        run_single(args, logger)

    logger.info("Inference complete!")


if __name__ == "__main__":
    main()
