"""Exports written to a private temporary directory and returned as data."""

from __future__ import annotations

import base64
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable


def export_files(write: Callable[[str], Any]) -> list[dict[str, str]]:
    """Run ``write`` on a fresh temporary directory and return what it wrote.

    Each file is ``{name, content_base64}``, with ``name`` relative to the
    directory. The directory is removed afterwards, so a caller never chooses
    where files land on the host.
    """
    with TemporaryDirectory() as directory:
        write(directory)
        root = Path(directory)
        return [
            {
                "name": path.relative_to(root).as_posix(),
                "content_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ]


__all__ = ["export_files"]
