"""Worker counts that the running Python can actually use.

CPython built for Emscripten (Pyodide, in a browser) cannot start threads, so
a thread pool there fails on its first submit. Callers that offer ``workers``
ask :func:`supported_workers` and take their serial path when it returns 1.
"""

from __future__ import annotations

import sys


def supported_workers(requested: int) -> int:
    """Return ``requested``, or 1 where the runtime cannot start threads."""
    if sys.platform == "emscripten":
        return 1
    return requested


__all__ = ["supported_workers"]
