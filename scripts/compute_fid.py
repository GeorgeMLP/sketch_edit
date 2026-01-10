"""Compute FID score between prediction and ground truth directories.

This script uses pytorch-fid to compute the Frechet Inception Distance.
Install it with: pip install pytorch-fid
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sketch_edit.utils import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute FID score between two image directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--predictions_dir",
        type=str,
        required=True,
        help="Directory containing predicted images.",
    )
    parser.add_argument(
        "--ground_truth_dir",
        type=str,
        required=True,
        help="Directory containing ground truth images.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Batch size for FID computation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use.",
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=2048,
        help="Dimensionality of Inception features.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/fid.log",
        help="Log file path.",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    setup_logging(log_file=args.log_file)
    logger = get_logger(__name__)

    # Verify directories exist
    pred_dir = Path(args.predictions_dir)
    gt_dir = Path(args.ground_truth_dir)

    if not pred_dir.exists():
        logger.error(f"Predictions directory not found: {pred_dir}")
        sys.exit(1)

    if not gt_dir.exists():
        logger.error(f"Ground truth directory not found: {gt_dir}")
        sys.exit(1)

    # Count images
    pred_count = len(list(pred_dir.glob("*.png"))) + len(list(pred_dir.glob("*.jpg")))
    gt_count = len(list(gt_dir.glob("*.png"))) + len(list(gt_dir.glob("*.jpg")))

    logger.info(f"Predictions: {pred_count} images")
    logger.info(f"Ground truth: {gt_count} images")

    if pred_count == 0 or gt_count == 0:
        logger.error("One or both directories are empty")
        sys.exit(1)

    # Run pytorch-fid
    logger.info("Computing FID score...")

    cmd = [
        sys.executable,
        "-m",
        "pytorch_fid",
        str(pred_dir),
        str(gt_dir),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--dims",
        str(args.dims),
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        fid_output = result.stdout.strip()
        logger.info(f"FID Output: {fid_output}")

        # Parse FID value
        if "FID:" in fid_output:
            fid_value = float(fid_output.split("FID:")[-1].strip())
            logger.info(f"FID Score: {fid_value:.4f}")
        else:
            # pytorch-fid outputs just the number
            fid_value = float(fid_output.split()[-1])
            logger.info(f"FID Score: {fid_value:.4f}")

    except subprocess.CalledProcessError as e:
        logger.error(f"FID computation failed: {e}")
        logger.error(f"stderr: {e.stderr}")
        sys.exit(1)
    except ImportError:
        logger.error("pytorch-fid not installed. Run: pip install pytorch-fid")
        sys.exit(1)


if __name__ == "__main__":
    main()
