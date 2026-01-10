"""Generate synthetic dataset for sketch+text-guided image editing."""

from __future__ import annotations

import argparse
from pathlib import Path

from sketch_edit.utils import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset for sketch+text-guided image editing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["instructions", "segmentation", "inpainting", "manifest", "all"],
        help="Task to run: generate instructions, segment objects, inpaint (remove objects), "
             "create manifest, or all steps.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing the dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to data_dir for most tasks.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="dataset.json",
        help="Output manifest file path (for 'manifest' task).",
    )

    # Instruction generation options
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/gemma-3-4b-it",
        help="HuggingFace model ID for instruction generation.",
    )

    # Segmentation options
    parser.add_argument(
        "--dino_model_id",
        type=str,
        default="IDEA-Research/grounding-dino-tiny",
        help="Grounding DINO model ID.",
    )
    parser.add_argument(
        "--sam_model_id",
        type=str,
        default="facebook/sam-vit-huge",
        help="SAM model ID.",
    )

    # Inpainting options
    parser.add_argument(
        "--inpaint_model_id",
        type=str,
        default="runwayml/stable-diffusion-inpainting",
        help="Inpainting model ID.",
    )
    parser.add_argument(
        "--use_existing_inpainted",
        action="store_true",
        help="Use existing inpainted images instead of generating new ones.",
    )
    parser.add_argument(
        "--existing_inpainted_dir",
        type=str,
        default=None,
        help="Directory containing existing inpainted images to copy.",
    )

    # Common options
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process. None for all.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip samples that already have outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run models on.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/generate_dataset.log",
        help="Log file path.",
    )

    return parser.parse_args()


def run_instructions(args: argparse.Namespace, logger) -> None:
    """Generate editing instructions using VLM."""
    from sketch_edit.data.generation import InstructionGenerator

    output_dir = Path(args.output_dir or args.data_dir) / "instructions"
    logger.info(f"Generating instructions to {output_dir}")

    generator = InstructionGenerator(
        model_id=args.model_id,
        device=args.device,
    )

    generator.generate_instructions_batch(
        data_dir=args.data_dir,
        output_dir=output_dir,
        num_samples=args.num_samples,
        skip_existing=args.skip_existing,
    )


def run_segmentation(args: argparse.Namespace, logger) -> None:
    """Segment objects using Grounding DINO + SAM."""
    from sketch_edit.data.generation import GroundedSegmenter

    output_dir = Path(args.output_dir or args.data_dir) / "segmentation"
    logger.info(f"Running segmentation, output to {output_dir}")

    segmenter = GroundedSegmenter(
        dino_model_id=args.dino_model_id,
        sam_model_id=args.sam_model_id,
        device=args.device,
    )

    segmenter.segment_dataset(
        data_dir=args.data_dir,
        output_dir=output_dir,
        skip_existing=args.skip_existing,
    )


def run_inpainting(args: argparse.Namespace, logger) -> None:
    """Remove objects via inpainting."""
    output_dir = Path(args.output_dir or args.data_dir) / "inpainted"

    if args.use_existing_inpainted:
        from sketch_edit.data.generation.inpainting import copy_existing_inpainted

        source_dir = args.existing_inpainted_dir or Path(args.data_dir) / "inpainted"
        logger.info(f"Copying existing inpainted images from {source_dir}")
        copy_existing_inpainted(source_dir, output_dir)
    else:
        from sketch_edit.data.generation import ObjectInpainter

        segmentation_dir = Path(args.data_dir) / "segmentation"
        if not segmentation_dir.exists():
            logger.error("Segmentation not found. Run --task segmentation first.")
            return

        logger.info(f"Running inpainting, output to {output_dir}")

        inpainter = ObjectInpainter(
            model_id=args.inpaint_model_id,
            device=args.device,
        )

        inpainter.inpaint_dataset(
            data_dir=args.data_dir,
            segmentation_dir=segmentation_dir,
            output_dir=output_dir,
            skip_existing=args.skip_existing,
        )


def run_manifest(args: argparse.Namespace, logger) -> None:
    """Create dataset manifest JSON."""
    from sketch_edit.data.generation import create_dataset_manifest

    output_file = args.output_file
    if not Path(output_file).is_absolute():
        output_file = Path(args.data_dir) / output_file

    logger.info(f"Creating manifest at {output_file}")

    create_dataset_manifest(
        data_dir=args.data_dir,
        output_file=output_file,
        validate=True,
    )


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Set up logging
    setup_logging(log_file=args.log_file)
    logger = get_logger(__name__)

    logger.info(f"Running task: {args.task}")
    logger.info(f"Data directory: {args.data_dir}")

    if args.task == "instructions":
        run_instructions(args, logger)
    elif args.task == "segmentation":
        run_segmentation(args, logger)
    elif args.task == "inpainting":
        run_inpainting(args, logger)
    elif args.task == "manifest":
        run_manifest(args, logger)
    elif args.task == "all":
        logger.info("Running all dataset generation steps...")
        run_segmentation(args, logger)
        run_inpainting(args, logger)
        run_instructions(args, logger)
        run_manifest(args, logger)

    logger.info("Done!")


if __name__ == "__main__":
    main()
