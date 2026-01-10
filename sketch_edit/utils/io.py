"""I/O utilities for file operations."""

from __future__ import annotations

import json
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> dict | list:
    """
    Load JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed JSON content.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict | list, path: str | Path, indent: int = 2) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save.
        path: Output file path.
        indent: JSON indentation level.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_text_lines(path: str | Path) -> list[str]:
    """
    Load text file as list of lines.

    Args:
        path: Path to text file.

    Returns:
        List of stripped lines.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def save_text_lines(lines: list[str], path: str | Path) -> None:
    """
    Save list of strings to text file, one per line.

    Args:
        lines: Lines to save.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in lines)
