"""Evaluate model predictions against ground truth."""

from __future__ import annotations

import argparse

from sketch_edit.evaluation import Evaluator
from sketch_edit.utils import get_logger, save_json, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions against ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input paths
    parser.add_argument(
        "--predictions_dir",
        type=str,
        required=True,
        help="Directory containing predicted images ({id}.png).",
    )
    parser.add_argument(
        "--ground_truth_dir",
        type=str,
        default=None,
        help="Directory containing ground truth images. "
             "If not provided, uses --manifest for ground truth paths.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to dataset manifest JSON for ground truth paths.",
    )
    parser.add_argument(
        "--bboxes_file",
        type=str,
        default=None,
        help="Optional JSON file with bounding boxes for localized evaluation.",
    )

    # Metrics
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["mse", "hist_intersection", "clip_similarity"],
        help="Metrics to compute.",
    )

    # Output
    parser.add_argument(
        "--output_file",
        type=str,
        default="eval_results.json",
        help="Output file for results (JSON or CSV).",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default="eval_summary.json",
        help="Output file for summary statistics.",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for neural network metrics.",
    )

    # Logging
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/evaluate.log",
        help="Log file path.",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Set up logging
    setup_logging(log_file=args.log_file)
    logger = get_logger(__name__)

    logger.info("=" * 50)
    logger.info("Starting evaluation")
    logger.info("=" * 50)

    # Validate inputs
    if args.ground_truth_dir is None and args.manifest is None:
        logger.error("Must provide either --ground_truth_dir or --manifest")
        return

    # Create evaluator
    logger.info(f"Metrics: {args.metrics}")
    evaluator = Evaluator(
        metrics=args.metrics,
        device=args.device,
    )

    # Run evaluation
    if args.manifest:
        logger.info(f"Evaluating using manifest: {args.manifest}")
        df = evaluator.evaluate_manifest(
            manifest_path=args.manifest,
            predictions_dir=args.predictions_dir,
            output_file=args.output_file,
        )
    else:
        logger.info(f"Evaluating directory: {args.predictions_dir}")
        df = evaluator.evaluate_directory(
            predictions_dir=args.predictions_dir,
            targets_dir=args.ground_truth_dir,
            bboxes_file=args.bboxes_file,
            output_file=args.output_file,
        )

    # Compute and save summary
    summary = evaluator.compute_summary(df)
    save_json(summary, args.summary_file)

    logger.info("=" * 50)
    logger.info("Evaluation Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    logger.info("=" * 50)

    logger.info(f"Results saved to {args.output_file}")
    logger.info(f"Summary saved to {args.summary_file}")
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
