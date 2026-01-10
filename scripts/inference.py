"""Run inference with sketch+text-guided image editing models."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from sketch_edit.models import ModelRegistry
from sketch_edit.utils import get_logger, save_image, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with sketch+text-guided image editing models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="controlnet_inpaint",
        choices=ModelRegistry.list_models(),
        help="Model to use for inference.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint. If None, uses pretrained weights.",
    )

    # Single image inference
    parser.add_argument(
        "--original_image",
        type=str,
        default=None,
        help="Path to original image.",
    )
    parser.add_argument(
        "--sketch",
        type=str,
        default=None,
        help="Path to sketch image.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for editing.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="edited.png",
        help="Output path for edited image.",
    )

    # Batch inference
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to dataset manifest for batch inference.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./predictions",
        help="Output directory for batch predictions.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Maximum samples for batch inference.",
    )

    # Generation parameters
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--image_guidance_scale",
        type=float,
        default=1.5,
        help="Image guidance scale (for InstructPix2Pix).",
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.0,
        help="ControlNet conditioning scale.",
    )
    parser.add_argument(
        "--invert_sketch",
        action="store_true",
        default=True,
        help="Invert sketch colors.",
    )
    parser.add_argument(
        "--no_invert_sketch",
        action="store_false",
        dest="invert_sketch",
        help="Don't invert sketch colors.",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Model dtype.",
    )

    # Logging
    parser.add_argument(
        "--log_file",
        type=str,
        default="inference.log",
        help="Log file path.",
    )

    return parser.parse_args()


def get_dtype(dtype_str: str):
    """Convert string to torch dtype."""
    import torch

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_str]


def run_single_inference(args: argparse.Namespace, logger) -> None:
    """Run inference on a single image."""
    if not all([args.original_image, args.sketch, args.prompt]):
        logger.error("For single inference, provide --original_image, --sketch, and --prompt")
        return

    # Load model
    logger.info(f"Loading model: {args.model}")
    model = ModelRegistry.create(
        args.model,
        device=args.device,
        dtype=get_dtype(args.dtype),
    )
    model.load(checkpoint_path=args.checkpoint)

    # Load images
    logger.info(f"Loading images...")
    original = Image.open(args.original_image).convert("RGB")
    sketch = Image.open(args.sketch).convert("RGB")

    # Run inference
    logger.info(f"Running inference with prompt: {args.prompt}")
    edited = model.edit(
        original_image=original,
        sketch_image=sketch,
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        invert_sketch=args.invert_sketch,
    )

    # Save output
    save_image(edited, args.output)
    logger.info(f"Saved edited image to {args.output}")


def run_batch_inference(args: argparse.Namespace, logger) -> None:
    """Run inference on a batch of images from manifest."""
    from tqdm import tqdm

    from sketch_edit.utils import load_json

    # Load manifest
    manifest = load_json(args.manifest)
    if args.num_samples:
        manifest = manifest[:args.num_samples]

    logger.info(f"Processing {len(manifest)} samples")

    # Load model
    logger.info(f"Loading model: {args.model}")
    model = ModelRegistry.create(
        args.model,
        device=args.device,
        dtype=get_dtype(args.dtype),
    )
    model.load(checkpoint_path=args.checkpoint)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process samples
    for entry in tqdm(manifest, desc="Inference"):
        sample_id = entry["id"]
        output_path = output_dir / f"{sample_id}.png"

        if output_path.exists():
            continue

        try:
            original = Image.open(entry["original_image_path"]).convert("RGB")
            sketch = Image.open(entry["sketch_image_path"]).convert("RGB")
            prompt = entry["edit_text"]

            edited = model.edit(
                original_image=original,
                sketch_image=sketch,
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                image_guidance_scale=args.image_guidance_scale,
                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                invert_sketch=args.invert_sketch,
            )

            save_image(edited, output_path)

        except Exception as e:
            logger.error(f"Error processing {sample_id}: {e}")

    logger.info(f"Predictions saved to {output_dir}")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Set up logging
    setup_logging(log_file=args.log_file)
    logger = get_logger(__name__)

    logger.info("=" * 50)
    logger.info("Starting inference")
    logger.info("=" * 50)

    if args.manifest:
        run_batch_inference(args, logger)
    else:
        run_single_inference(args, logger)

    logger.info("Inference complete!")


if __name__ == "__main__":
    main()
