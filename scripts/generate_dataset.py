"""Generate the multi-task synthetic dataset for sketch+text-guided image editing."""

from __future__ import annotations

import argparse
from pathlib import Path

from sketch_edit.utils import get_logger, load_json, save_json, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset for sketch+text-guided image editing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["segmentation", "generate", "instructions", "manifest", "all"],
        help="Pipeline step to run.",
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Root dataset directory.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory.")

    # Editing-task selection (for 'generate' and 'instructions' steps)
    parser.add_argument(
        "--editing_tasks",
        type=str,
        nargs="+",
        default=None,
        help="Which editing tasks to process (default: all registered tasks).",
    )

    # Segmentation models
    parser.add_argument("--dino_model_id", type=str, default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam_model_id", type=str, default="facebook/sam-vit-huge")

    # Inpainting model
    parser.add_argument("--lama_ckpt_dir", type=str, default="pretrained_models/big-lama")
    parser.add_argument("--dilate_kernel_size", type=int, default=15)

    # VLM instruction model
    parser.add_argument("--model_id", type=str, default="google/gemma-3-4b-it")

    # Manifest / splits
    parser.add_argument("--output_file", type=str, default="dataset.json")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # Common
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_file", type=str, default="logs/generate_dataset.log")

    return parser.parse_args()


def run_segmentation(args: argparse.Namespace, logger) -> None:
    from sketch_edit.data.generation import GroundedSegmenter

    output_dir = Path(args.output_dir or args.data_dir) / "segmentation"
    logger.info(f"Running segmentation → {output_dir}")

    segmenter = GroundedSegmenter(
        dino_model_id=args.dino_model_id,
        sam_model_id=args.sam_model_id,
        device=args.device,
    )
    segmenter.segment_dataset(
        data_dir=args.data_dir,
        output_dir=output_dir,
        skip_existing=args.skip_existing,
        num_samples=args.num_samples,
    )


def run_generate(args: argparse.Namespace, logger) -> None:
    """Generate task-specific input images."""
    from sketch_edit.data.generation import LamaInpainter, get_task, list_tasks

    data_dir = Path(args.data_dir)
    seg_dir = data_dir / "segmentation"
    generated_dir = Path(args.output_dir or args.data_dir) / "generated"

    # Which tasks to run
    task_names = args.editing_tasks or list_tasks()
    logger.info(f"Generating inputs for tasks: {task_names}")

    # Load segmentation results
    seg_results_path = seg_dir / "segmentation_results.json"
    if not seg_results_path.exists():
        logger.error("Segmentation results not found. Run --step segmentation first.")
        return
    seg_results = load_json(seg_results_path)

    # Collect sample IDs
    sampled_dir = data_dir / "sampled"
    sample_ids = sorted(d.name for d in sampled_dir.iterdir() if d.is_dir())
    if args.num_samples is not None:
        sample_ids = sample_ids[: args.num_samples]

    # Only initialize inpainter if at least one task needs it
    tasks = [get_task(n) for n in task_names]
    inpainter = None
    if any(t.requires_inpainter for t in tasks):
        logger.info("Loading LaMa inpainter (shared across tasks)…")
        inpainter = LamaInpainter(lama_ckpt_dir=args.lama_ckpt_dir, device=args.device)

    for task in tasks:
        task_dir = generated_dir / task.name
        logger.info(f"--- Task: {task.name} ({task.description}) ---")

        results = task.generate_inputs(
            data_dir=data_dir,
            segmentation_dir=seg_dir,
            output_dir=task_dir,
            inpainter=inpainter,
            sample_ids=sample_ids,
            seg_results=seg_results,
            skip_existing=args.skip_existing,
        )

        # Persist metadata so the manifest step can pick it up
        save_json(results, task_dir / "metadata.json")
        logger.info(f"  {task.name}: {len(results)} samples → {task_dir}")


def run_instructions(args: argparse.Namespace, logger) -> None:
    """Generate VLM instructions for each task."""
    from sketch_edit.data.generation import InstructionGenerator, list_tasks

    data_dir = Path(args.data_dir)
    task_names = args.editing_tasks or list_tasks()

    generator = InstructionGenerator(model_id=args.model_id, device=args.device)

    for task_name in task_names:
        out_dir = data_dir / "instructions" / task_name
        logger.info(f"VLM instructions for task '{task_name}' → {out_dir}")
        generator.generate_instructions_batch(
            data_dir=data_dir,
            output_dir=out_dir,
            task_name=task_name,
            num_samples=args.num_samples,
            skip_existing=args.skip_existing,
        )


def run_manifest(args: argparse.Namespace, logger) -> None:
    from sketch_edit.data.generation import create_dataset_manifest, list_tasks

    data_dir = Path(args.data_dir)
    output_file = args.output_file
    if not Path(output_file).is_absolute():
        output_file = data_dir / output_file

    task_names = args.editing_tasks or list_tasks()
    logger.info(f"Creating manifest at {output_file}")

    create_dataset_manifest(
        data_dir=data_dir,
        output_file=output_file,
        task_names=task_names,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        validate=True,
    )


def main() -> None:
    args = parse_args()
    setup_logging(log_file=args.log_file)
    logger = get_logger(__name__)

    logger.info(f"Step: {args.step} | Data dir: {args.data_dir}")

    if args.step in ("segmentation", "all"):
        run_segmentation(args, logger)
    if args.step in ("generate", "all"):
        run_generate(args, logger)
    if args.step in ("instructions",):
        run_instructions(args, logger)
    if args.step in ("manifest", "all"):
        run_manifest(args, logger)

    logger.info("Done!")


if __name__ == "__main__":
    main()
