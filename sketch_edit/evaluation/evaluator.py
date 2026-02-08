"""Evaluator for batch evaluation of model predictions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from sketch_edit.evaluation.metrics import (
    compute_clip_similarity,
    compute_histogram_intersection,
    compute_mse,
)
from sketch_edit.utils import get_logger, load_json, save_json

logger = get_logger(__name__)


class Evaluator:
    """
    Batch evaluator for model predictions.

    Computes metrics across a set of prediction-target pairs.
    """

    def __init__(
        self,
        metrics: list[str] | None = None,
        device: str | torch.device = "cuda",
    ):
        """
        Initialize evaluator.

        Args:
            metrics: List of metrics to compute. Default: ["mse", "hist_intersection", "clip_similarity"].
            device: Device for neural network metrics.
        """
        self.metrics = metrics or ["mse", "hist_intersection", "clip_similarity"]
        self.device = device

        # Pre-load models for efficiency
        self._clip_model = None
        self._lpips_fn = None

        if "clip_similarity" in self.metrics:
            self._load_clip_model()

        if "lpips" in self.metrics:
            self._load_lpips_fn()

    def _load_clip_model(self) -> None:
        """Load CLIP model."""
        from transformers import CLIPModel

        logger.info("Loading CLIP model...")
        self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self._clip_model = self._clip_model.to(self.device)
        self._clip_model.eval()

    def _load_lpips_fn(self) -> None:
        """Load LPIPS function."""
        try:
            import lpips

            logger.info("Loading LPIPS model...")
            self._lpips_fn = lpips.LPIPS(net="vgg").to(self.device)
            self._lpips_fn.eval()
        except ImportError:
            logger.warning("lpips not installed, skipping LPIPS metric")
            self.metrics = [m for m in self.metrics if m != "lpips"]

    def evaluate_pair(
        self,
        pred_image: Image.Image,
        target_image: Image.Image,
        bbox: list[int] | None = None,
    ) -> dict[str, float]:
        """
        Evaluate a single prediction-target pair.

        Args:
            pred_image: Predicted image.
            target_image: Ground truth image.
            bbox: Optional bounding box [x1, y1, x2, y2] to crop before evaluation.

        Returns:
            Dictionary of metric values.
        """
        import numpy as np

        pred = np.array(pred_image.convert("RGB"))
        target = np.array(target_image.convert("RGB"))

        # Crop to bounding box if provided
        if bbox is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            pred = pred[y1:y2, x1:x2]
            target = target[y1:y2, x1:x2]

        results = {}

        if "mse" in self.metrics:
            results["mse"] = compute_mse(pred, target)

        if "hist_intersection" in self.metrics:
            results["hist_intersection"] = compute_histogram_intersection(pred, target)

        if "clip_similarity" in self.metrics:
            results["clip_similarity"] = compute_clip_similarity(
                pred, target, model=self._clip_model, device=self.device
            )

        if "lpips" in self.metrics and self._lpips_fn is not None:
            from sketch_edit.evaluation.metrics import compute_lpips

            results["lpips"] = compute_lpips(
                pred, target, loss_fn=self._lpips_fn, device=self.device
            )

        return results

    def evaluate_directory(
        self,
        predictions_dir: str | Path,
        targets_dir: str | Path,
        bboxes_file: str | Path | None = None,
        output_file: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        Evaluate all predictions in a directory against ground truth.

        Expects files named {id}.png in both directories.

        Args:
            predictions_dir: Directory containing predicted images.
            targets_dir: Directory containing ground truth images.
            bboxes_file: Optional JSON file with bounding boxes.
            output_file: Optional path to save results CSV.

        Returns:
            DataFrame with per-sample metrics.
        """
        predictions_dir = Path(predictions_dir)
        targets_dir = Path(targets_dir)

        # Load bounding boxes if provided
        bboxes = {}
        if bboxes_file and Path(bboxes_file).exists():
            raw_bboxes = load_json(bboxes_file)
            for id_, entry in raw_bboxes.items():
                if isinstance(entry, dict):
                    bboxes[id_] = entry.get("bbox")
                else:
                    bboxes[id_] = entry

        # Find prediction files
        pred_files = sorted(predictions_dir.glob("*.png")) + sorted(predictions_dir.glob("*.jpg"))
        logger.info(f"Found {len(pred_files)} prediction files")

        results = []
        for pred_path in tqdm(pred_files, desc="Evaluating"):
            sample_id = pred_path.stem
            target_path = targets_dir / pred_path.name

            if not target_path.exists():
                # Try different extension
                for ext in [".png", ".jpg", ".jpeg"]:
                    alt_path = targets_dir / f"{sample_id}{ext}"
                    if alt_path.exists():
                        target_path = alt_path
                        break

            if not target_path.exists():
                logger.warning(f"No target found for {sample_id}")
                continue

            try:
                pred_img = Image.open(pred_path)
                target_img = Image.open(target_path)

                bbox = bboxes.get(sample_id)
                metrics = self.evaluate_pair(pred_img, target_img, bbox=bbox)

                metrics["id"] = sample_id
                results.append(metrics)

            except Exception as e:
                logger.error(f"Error evaluating {sample_id}: {e}")

        df = pd.DataFrame(results)

        # Compute summary statistics
        if len(df) > 0:
            logger.info(f"Evaluated {len(df)} samples")
            for metric in self.metrics:
                if metric in df.columns:
                    logger.info(f"  {metric}: {df[metric].mean():.4f} ± {df[metric].std():.4f}")

        # Save results
        if output_file is not None:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if output_file.suffix == ".csv":
                df.to_csv(output_file, index=False)
            elif output_file.suffix == ".json":
                save_json(df.to_dict(orient="records"), output_file)

            logger.info(f"Results saved to {output_file}")

        return df

    def evaluate_manifest(
        self,
        manifest_path: str | Path,
        predictions_dir: str | Path,
        split: str = "test",
        task: str | None = None,
        output_file: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        Evaluate predictions using a dataset manifest.

        Args:
            manifest_path: Path to dataset manifest JSON.
            predictions_dir: Directory containing predicted images.
            split: Which split to evaluate (train/val/test).
            task: Optional task filter.
            output_file: Optional path to save results.

        Returns:
            DataFrame with per-sample metrics.
        """
        raw = load_json(manifest_path)
        # Support both split-aware (dict with train/val/test) and flat (list) manifests
        if isinstance(raw, dict) and split in raw:
            entries = raw[split]
        elif isinstance(raw, list):
            entries = raw
        else:
            entries = raw.get("test", [])

        if task is not None:
            entries = [e for e in entries if e.get("task") == task]

        predictions_dir = Path(predictions_dir)

        results = []
        for entry in tqdm(entries, desc="Evaluating"):
            sample_id = entry["id"]
            pred_path = predictions_dir / f"{sample_id}.png"

            if not pred_path.exists():
                pred_path = predictions_dir / f"{sample_id}.jpg"

            if not pred_path.exists():
                continue

            target_path = Path(entry["target_image_path"])
            if not target_path.exists():
                continue

            try:
                pred_img = Image.open(pred_path)
                target_img = Image.open(target_path)

                metrics = self.evaluate_pair(pred_img, target_img)
                metrics["id"] = sample_id
                metrics["edit_text"] = entry.get("edit_text", "")
                results.append(metrics)

            except Exception as e:
                logger.error(f"Error evaluating {sample_id}: {e}")

        df = pd.DataFrame(results)

        # Log summary
        if len(df) > 0:
            logger.info(f"Evaluated {len(df)} samples")
            for metric in self.metrics:
                if metric in df.columns:
                    logger.info(f"  {metric}: {df[metric].mean():.4f} ± {df[metric].std():.4f}")

        # Save results
        if output_file is not None:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if output_file.suffix == ".csv":
                df.to_csv(output_file, index=False)
            else:
                save_json(df.to_dict(orient="records"), output_file)

            logger.info(f"Results saved to {output_file}")

        return df

    def compute_summary(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Compute summary statistics from evaluation results.

        Args:
            df: DataFrame from evaluate_* methods.

        Returns:
            Dictionary of summary statistics.
        """
        summary = {"num_samples": len(df)}

        for metric in self.metrics:
            if metric in df.columns:
                summary[f"{metric}_mean"] = float(df[metric].mean())
                summary[f"{metric}_std"] = float(df[metric].std())
                summary[f"{metric}_median"] = float(df[metric].median())

        return summary
