"""Upload and download dataset to/from Weights & Biases."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import wandb

from sketch_edit.data.generation import create_dataset_manifest, list_tasks
from sketch_edit.utils import get_logger, setup_logging

logger = get_logger(__name__)


def upload_dataset(
    dataset_dir: str | Path,
    artifact_name: str = "dataset",
    description: str = "Sketch-guided image editing dataset",
) -> None:
    """
    Upload dataset to Weights & Biases.
    
    Args:
        dataset_dir: Path to dataset directory (contains sampled/, segmentation/, generated/, instructions/).
        artifact_name: Name for the wandb artifact.
        description: Description for the artifact.
    """
    dataset_dir = Path(dataset_dir).resolve()
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    logger.info(f"Preparing to upload dataset from {dataset_dir}")
    logger.info("Compressing dataset...")
    
    current_dir = Path.cwd()
    
    # Create tarball in a temp location
    temp_dir = dataset_dir.parent
    tar_filename = "dataset_temp.tar"
    compressed_filename = "dataset_temp.tar.bz2"
    
    try:
        # Create tarball of dataset contents (not the directory itself)
        # This way extraction creates sampled/, segmentation/, generated/, etc. directly
        logger.info(f"Creating tarball: {tar_filename}")
        os.chdir(dataset_dir)
        os.system(f"tar -cf ../{tar_filename} .")
        
        logger.info(f"Compressing with bzip2: {compressed_filename}")
        os.chdir(temp_dir)
        os.system(f"pbzip2 -f {tar_filename}")  # -f to overwrite if exists
        
        # Initialize wandb
        logger.info("Initializing wandb...")
        wandb.init(
            entity="sketch_edit",
            project="sketch_edit",
            job_type="upload-dataset",
        )
        
        # Create and upload artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type="dataset",
            description=description,
            metadata={
                "num_subdirs": len([d for d in dataset_dir.iterdir() if d.is_dir()]),
            },
        )
        
        compressed_file = temp_dir / compressed_filename
        artifact.add_file(str(compressed_file))
        logger.info(f"Uploading {artifact.name} to W&B...")
        wandb.log_artifact(artifact)
        
        logger.info("Upload complete!")
        
        # Clean up
        logger.info("Cleaning up temporary files...")
        if compressed_file.exists():
            os.remove(compressed_file)
        if (temp_dir / "wandb").exists():
            shutil.rmtree(temp_dir / "wandb")
            
    finally:
        os.chdir(current_dir)


def download_dataset(
    output_dir: str | Path,
    artifact_name: str = "dataset:latest",
    regenerate_manifest: bool = True,
) -> None:
    """
    Download and extract dataset from Weights & Biases.
    
    Args:
        output_dir: Directory to extract dataset to (will contain sampled/, segmentation/, generated/, etc.).
        artifact_name: Wandb artifact name (e.g., 'dataset:latest' or 'dataset:v1').
        regenerate_manifest: Whether to regenerate dataset.json with local paths.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading dataset to {output_dir}")
    
    current_dir = Path.cwd()
    os.chdir(output_dir)
    
    try:
        # Initialize wandb and download
        logger.info("Initializing wandb...")
        wandb.init(
            entity="sketch_edit",
            project="sketch_edit",
            job_type="download-dataset",
        )
        
        logger.info(f"Downloading artifact: {artifact_name}")
        artifact = wandb.use_artifact(artifact_name)
        artifact_dir = artifact.download(str(output_dir))
        
        # Extract the compressed file directly into output_dir
        logger.info("Extracting dataset...")
        compressed_file = list(Path(artifact_dir).glob("*.tar.bz2"))[0]
        
        logger.info(f"Decompressing {compressed_file.name}...")
        os.system(f"tar -xjf {compressed_file}")
        
        # The dataset is now directly in output_dir (sampled/, segmentation/, generated/, etc.)
        # Regenerate manifest with correct local paths
        if regenerate_manifest:
            logger.info("Regenerating manifest with local paths...")
            task_names = list_tasks()
            create_dataset_manifest(
                data_dir=output_dir,
                output_file=output_dir / "dataset.json",
                task_names=task_names,
                validate=False,  # Skip validation since paths are being reconstructed
            )
            logger.info("Manifest regenerated successfully!")
        
        logger.info("Download complete!")
        
        # Clean up
        logger.info("Cleaning up temporary files...")
        if compressed_file.exists():
            os.remove(compressed_file)
        if Path("wandb").exists():
            shutil.rmtree("wandb")
        if Path("artifacts").exists():
            shutil.rmtree("artifacts")
            
    finally:
        os.chdir(current_dir)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload/download dataset to/from Weights & Biases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--action",
        type=str,
        required=True,
        choices=["upload", "download"],
        help="Action to perform: upload or download dataset.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./dataset",
        help="Path to dataset directory (for upload) or output directory (for download).",
    )
    parser.add_argument(
        "--artifact_name",
        type=str,
        default=None,
        help="Wandb artifact name. For upload: artifact name (default: 'dataset'). "
             "For download: artifact with version (default: 'dataset:latest').",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="Sketch-guided image editing dataset",
        help="Description for the artifact (upload only).",
    )
    parser.add_argument(
        "--no_regenerate_manifest",
        action="store_true",
        help="Skip regenerating dataset.json after download.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/wandb_sync.log",
        help="Log file path.",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Set up logging
    setup_logging(log_file=args.log_file)
    
    if args.action == "upload":
        artifact_name = args.artifact_name or "dataset"
        upload_dataset(
            dataset_dir=args.dataset_dir,
            artifact_name=artifact_name,
            description=args.description,
        )
    elif args.action == "download":
        artifact_name = args.artifact_name or "dataset:latest"
        download_dataset(
            output_dir=args.dataset_dir,
            artifact_name=artifact_name,
            regenerate_manifest=not args.no_regenerate_manifest,
        )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
