"""Count Pint quantity constructions, to pin the cost of domain hot paths."""

from collections.abc import Iterator
from contextlib import contextmanager
from unittest import mock

from OpenPinch.domain.value import Q_


@contextmanager
def count_quantities() -> Iterator[list[int]]:
    """Yield a one-item list holding the quantities built inside the block."""
    built = [0]
    original = Q_.__new__

    def counting(cls, *args, **kwargs):
        built[0] += 1
        return original(cls, *args, **kwargs)

    with mock.patch.object(Q_, "__new__", counting):
        yield built
