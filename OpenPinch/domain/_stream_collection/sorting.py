"""Helper utilities for stream collection operations."""

from __future__ import annotations

import pickle
from typing import Tuple

from ..value import Value


def _sort_by_attr(attr: str, stream: object):
    return getattr(stream, attr)


def _sort_by_attrs(attrs: Tuple[str, ...], stream: object):
    return tuple(getattr(stream, attr) for attr in attrs)


def _stream_attr_value(stream: object, attr_name: str, idx: int | None = None):
    value = getattr(stream, attr_name)
    if isinstance(value, Value):
        period_idx = 0 if idx is None else int(idx)
        return float(value[period_idx])
    return value


def _is_picklable(obj: object) -> bool:
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False
