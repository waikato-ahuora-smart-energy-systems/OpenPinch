"""UTF-8 JSON filesystem primitives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_json(text: str) -> Any:
    """Parse one JSON string without imposing a domain schema."""
    return json.loads(text)


def read_json(path: str | Path) -> Any:
    """Read and parse one UTF-8 JSON document."""
    return parse_json(Path(path).read_text(encoding="utf-8"))


def write_json(
    path: str | Path,
    value: Any,
    *,
    indent: int = 4,
) -> Path:
    """Serialize one value to a UTF-8 JSON document."""
    destination = Path(path)
    destination.write_text(
        json.dumps(value, indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )
    return destination


__all__ = ["parse_json", "read_json", "write_json"]
