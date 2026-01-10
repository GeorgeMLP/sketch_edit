"""Train sketch+text-guided image editing models."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import CLIPTokenizer

from sketch_edit.data import create_dataloaders
from sketch_edit.training import Trainer, TrainingConfig
from sketch_edit.utils import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train sketch+text-guided image editing models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--dataset_json",
        type=str,
        required=True,
        help="Path to dataset manifest JSON.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of data for validation.",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="controlnet_inpaint",
        choices=["controlnet_scribble", "controlnet_inpaint", "instruct_pix2pix"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--pretrained_model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base Stable Diffusion model ID.",
    )
    parser.add_argument(
        "--controlnet_id",
        type=str,
        default="lllyasviel/control_v11p_sd15_scribble",
        help="ControlNet model ID.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps. Overrides num_epochs if set.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.1,
        help="Probability of dropping text conditioning.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Image size for training.",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints.",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps.",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=10,
        help="Log every N steps.",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train on.",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        default=True,
        help="Use mixed precision training.",
    )
    parser.add_argument(
        "--no_mixed_precision",
        action="store_false",
        dest="mixed_precision",
        help="Disable mixed precision training.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    # Logging
    parser.add_argument(
        "--log_file",
        type=str,
        default="train.log",
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
    logger.info("Starting training")
    logger.info("=" * 50)
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 50)

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_id, subfolder="tokenizer")

    # Create dataloaders
    logger.info(f"Loading dataset from {args.dataset_json}")
    train_loader, val_loader = create_dataloaders(
        manifest_path=args.dataset_json,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create training config
    config = TrainingConfig(
        model_name=args.model,
        pretrained_model_id=args.pretrained_model_id,
        controlnet_id=args.controlnet_id,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        conditioning_dropout_prob=args.conditioning_dropout_prob,
        image_size=args.image_size,
        val_split=args.val_split,
        output_dir=args.output_dir,
        save_every_n_steps=args.save_every_n_steps,
        log_every_n_steps=args.log_every_n_steps,
        mixed_precision=args.mixed_precision,
        device=args.device,
        seed=args.seed,
    )

    # Create trainer
    trainer = Trainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Resume from checkpoint if specified
    if args.resume_from:
        logger.info(f"Resuming from {args.resume_from}")
        trainer.load_checkpoint(Path(args.resume_from))

    # Train
    trainer.train()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
