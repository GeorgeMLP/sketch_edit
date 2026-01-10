"""Create dataset manifest from directory structure."""

from __future__ import annotations

from pathlib import Path

from sketch_edit.utils import get_logger, save_json, load_text_lines

logger = get_logger(__name__)


def create_dataset_manifest(
    data_dir: str | Path,
    output_file: str | Path,
    validate: bool = True,
) -> list[dict]:
    """
    Create a dataset manifest JSON from the directory structure.

    Expects directory structure:
    - data_dir/sampled/{id}/*_photo.jpg (ground truth with object)
    - data_dir/sampled/{id}/*_sketch.png (object sketch)
    - data_dir/inpainted/{id}/*_remove.jpg (background without object)
    - data_dir/instructions/{id}.txt (editing instructions)

    Args:
        data_dir: Root directory containing the dataset.
        output_file: Path to save the manifest JSON.
        validate: Whether to validate that all files exist.

    Returns:
        List of dataset entries.
    """
    data_dir = Path(data_dir)
    output_file = Path(output_file)

    sampled_dir = data_dir / "sampled"
    inpainted_dir = data_dir / "inpainted"
    instructions_dir = data_dir / "instructions"

    if not sampled_dir.exists():
        raise FileNotFoundError(f"Sampled directory not found: {sampled_dir}")

    # Find all sample directories
    sample_ids = sorted([d.name for d in sampled_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(sample_ids)} sample directories")

    dataset = []
    skipped = 0

    for sample_id in sample_ids:
        sample_dir = sampled_dir / sample_id
        inpainted_sample_dir = inpainted_dir / sample_id
        instruction_file = instructions_dir / f"{sample_id}.txt"

        # Find files
        photo_files = list(sample_dir.glob("*_photo.jpg")) + list(sample_dir.glob("*_photo.png"))
        sketch_files = list(sample_dir.glob("*_sketch.png")) + list(sample_dir.glob("*_sketch.jpg"))
        remove_files = (
            list(inpainted_sample_dir.glob("*_remove.jpg"))
            + list(inpainted_sample_dir.glob("*_remove.png"))
            if inpainted_sample_dir.exists()
            else []
        )

        # Validate files exist
        if validate:
            if not photo_files:
                logger.debug(f"Missing photo for sample {sample_id}")
                skipped += 1
                continue
            if not sketch_files:
                logger.debug(f"Missing sketch for sample {sample_id}")
                skipped += 1
                continue
            if not remove_files:
                logger.debug(f"Missing inpainted image for sample {sample_id}")
                skipped += 1
                continue
            if not instruction_file.exists():
                logger.debug(f"Missing instruction for sample {sample_id}")
                skipped += 1
                continue

        # Load instructions
        try:
            instructions = load_text_lines(instruction_file)
        except Exception:
            instructions = []

        if not instructions:
            logger.debug(f"Empty instruction for sample {sample_id}")
            skipped += 1
            continue

        # Create entries for each instruction
        for idx, instruction in enumerate(instructions):
            entry = {
                "id": f"{sample_id}_{idx}",
                "original_image_path": str(remove_files[0].absolute()) if remove_files else "",
                "sketch_image_path": str(sketch_files[0].absolute()) if sketch_files else "",
                "target_image_path": str(photo_files[0].absolute()) if photo_files else "",
                "edit_text": instruction,
            }
            dataset.append(entry)

    logger.info(f"Created manifest with {len(dataset)} entries ({skipped} samples skipped)")

    # Save manifest
    save_json(dataset, output_file)
    logger.info(f"Manifest saved to {output_file}")

    return dataset


def validate_manifest(manifest_path: str | Path) -> tuple[int, int]:
    """
    Validate that all files in a manifest exist.

    Args:
        manifest_path: Path to manifest JSON file.

    Returns:
        Tuple of (valid_count, invalid_count).
    """
    from sketch_edit.utils import load_json

    manifest = load_json(manifest_path)
    valid = 0
    invalid = 0

    for entry in manifest:
        all_exist = True
        for key in ["original_image_path", "sketch_image_path", "target_image_path"]:
            path = entry.get(key, "")
            if path and not Path(path).exists():
                logger.warning(f"Missing file: {path}")
                all_exist = False

        if all_exist:
            valid += 1
        else:
            invalid += 1

    logger.info(f"Validation complete: {valid} valid, {invalid} invalid")
    return valid, invalid
