import json
import logging
from pathlib import Path
from typing import Any, Iterable, Generator, Optional

from src.utils.logging_config import logger

def safe_read_jsonl(file_path: Path) -> Generator[dict, None, None]:
    """
    Robustly reads a JSONL file, skipping malformed lines and handling encoding.

    Args:
        file_path: Path to the JSONL file.

    Yields:
        Parsed JSON objects from each valid line.
    """
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.error(f"Malformed JSON on line {line_num} in {file_path}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Critical error reading {file_path}: {e}")

def safe_write_jsonl(file_path: Path, data: Iterable[dict], append: bool = False) -> None:
    """
    Robustly writes a list of dictionaries to a JSONL file.

    Args:
        file_path: Path to the output JSONL file.
        data: Iterable of dictionaries to write.
        append: If True, appends to the file; otherwise overwrites.
    """
    mode = "a" if append else "w"
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, mode, encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Critical error writing to {file_path}: {e}")
        raise
