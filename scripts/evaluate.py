"""Evaluate model predictions against ground truth."""

from __future__ import annotations

import argparse

from sketch_edit.evaluation import Evaluator
from sketch_edit.utils import get_logger, save_json, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions against ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--ground_truth_dir", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument(
        "--split", type=str, default="test",
        help="Which manifest split to evaluate (train/val/test).",
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Filter evaluation to a specific editing task.",
    )
    parser.add_argument("--bboxes_file", type=str, default=None)
    parser.add_argument(
        "--metrics", type=str, nargs="+",
        default=["mse", "hist_intersection", "clip_similarity"],
    )

    parser.add_argument("--output_file", type=str, default="eval_results.json")
    parser.add_argument("--summary_file", type=str, default="eval_summary.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_file", type=str, default="logs/evaluate.log")

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)
    logger = get_logger(__name__)

    logger.info("=" * 50)
    logger.info("Starting evaluation")
    logger.info("=" * 50)

    if args.ground_truth_dir is None and args.manifest is None:
        logger.error("Must provide either --ground_truth_dir or --manifest")
        return

    evaluator = Evaluator(metrics=args.metrics, device=args.device)

    if args.manifest:
        logger.info(f"Evaluating split='{args.split}' from {args.manifest}")
        df = evaluator.evaluate_manifest(
            manifest_path=args.manifest,
            predictions_dir=args.predictions_dir,
            split=args.split,
            task=args.task,
            output_file=args.output_file,
        )
    else:
        df = evaluator.evaluate_directory(
            predictions_dir=args.predictions_dir,
            targets_dir=args.ground_truth_dir,
            bboxes_file=args.bboxes_file,
            output_file=args.output_file,
        )

    summary = evaluator.compute_summary(df)
    save_json(summary, args.summary_file)

    logger.info("=" * 50)
    logger.info("Evaluation Summary:")
    for k, v in summary.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    logger.info("=" * 50)
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
