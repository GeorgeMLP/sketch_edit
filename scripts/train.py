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
    parser = argparse.ArgumentParser(
        description="Train sketch+text-guided image editing models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--manifest", type=str, required=True, help="Path to dataset manifest JSON.")
    parser.add_argument(
        "--task", type=str, default=None,
        help="Train on a specific editing task only (e.g. 'object_addition'). Default: all tasks.",
    )

    # Model
    parser.add_argument(
        "--model", type=str, default="controlnet_inpaint",
        choices=["controlnet_scribble", "controlnet_inpaint", "instruct_pix2pix"],
    )
    parser.add_argument("--pretrained_model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet_id", type=str, default="lllyasviel/control_v11p_sd15_scribble")
    parser.add_argument("--resume_from", type=str, default=None)

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--conditioning_dropout_prob", type=float, default=0.1)
    parser.add_argument("--image_size", type=int, default=512)

    # Output
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_every_n_steps", type=int, default=500)
    parser.add_argument("--log_every_n_steps", type=int, default=10)

    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", action="store_true", default=True)
    parser.add_argument("--no_mixed_precision", action="store_false", dest="mixed_precision")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--log_file", type=str, default="logs/train.log")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)
    logger = get_logger(__name__)

    logger.info("=" * 50)
    logger.info("Starting training")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 50)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_id, subfolder="tokenizer")

    train_loader, val_loader = create_dataloaders(
        manifest_path=args.manifest,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        image_size=args.image_size,
        task=args.task,
        num_workers=args.num_workers,
    )

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
        output_dir=args.output_dir,
        save_every_n_steps=args.save_every_n_steps,
        log_every_n_steps=args.log_every_n_steps,
        mixed_precision=args.mixed_precision,
        device=args.device,
        seed=args.seed,
    )

    trainer = Trainer(config=config, train_loader=train_loader, val_loader=val_loader)

    if args.resume_from:
        trainer.load_checkpoint(Path(args.resume_from))

    trainer.train()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
