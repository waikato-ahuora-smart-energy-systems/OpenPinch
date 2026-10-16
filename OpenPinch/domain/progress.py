"""Progress reports from long-running analyses.

An analysis calls :func:`emit` as it works; a caller that wants the reports
wraps the analysis in :func:`reporting`. Outside :func:`reporting`, :func:`emit`
returns at once, so reporting never changes a result.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Progress:
    """One progress report from a long-running analysis."""

    analysis: str
    stage: str
    message: str
    done: int | None
    total: int | None
    unit: str | None
    best_objective: float | None
    objective: str | None
    elapsed: float
    scope: str | None

    def to_json(self) -> dict[str, Any]:
        """The report as strict JSON data: NaN and infinities become ``None``."""
        data = asdict(self)
        best = data["best_objective"]
        if best is not None and not math.isfinite(best):
            data["best_objective"] = None
        return data


class _Reporter:
    def __init__(
        self,
        callback: Callable[[Progress], None],
        min_interval: float,
        clock: Callable[[], float],
    ) -> None:
        self.callback = callback
        self.min_interval = min_interval
        self.clock = clock
        self.start = clock()
        # A forked worker process inherits the reporter; only this one reports.
        self.pid = os.getpid()
        # When each stage last reached the callback; threads share it.
        self.forwarded: dict[tuple[str, str, str | None], float] = {}
        self.lock = threading.Lock()

    def emit(self, **fields) -> None:
        if os.getpid() != self.pid:
            return
        now = self.clock()
        scope = _SCOPE.get()
        key = (fields["analysis"], fields["stage"], scope)
        finished = fields["total"] is not None and fields["done"] == fields["total"]
        with self.lock:
            last = self.forwarded.get(key)
            if not finished and last is not None and now - last < self.min_interval:
                return
            self.forwarded[key] = now
        detail = fields.pop("detail")
        self.callback(
            Progress(
                message=_message(**fields, detail=detail),
                elapsed=now - self.start,
                scope=scope,
                **fields,
            )
        )


_REPORTER: ContextVar[_Reporter | None] = ContextVar(
    "openpinch_progress_reporter", default=None
)

# The part of a problem the enclosed analysis works on, such as a period.
_SCOPE: ContextVar[str | None] = ContextVar("openpinch_progress_scope", default=None)


@contextmanager
def reporting(
    callback: Callable[[Progress], None],
    *,
    min_interval: float = 0.25,
    clock: Callable[[], float] = time.monotonic,
) -> Iterator[None]:
    """Pass the progress the enclosed analyses report to ``callback``.

    Reports of one stage closer together than ``min_interval`` seconds are
    dropped, except a stage's first report and the one that finishes it
    (``done == total``). An exception the callback raises propagates out of
    the analysis, so a callback can cancel it.
    """
    token = _REPORTER.set(_Reporter(callback, min_interval, clock))
    try:
        yield
    finally:
        _REPORTER.reset(token)


@contextmanager
def scope(name: str) -> Iterator[None]:
    """Label the progress the enclosed analysis reports with ``name``."""
    token = _SCOPE.set(name)
    try:
        yield
    finally:
        _SCOPE.reset(token)


def emit(
    *,
    analysis: str,
    stage: str,
    done: int | None = None,
    total: int | None = None,
    unit: str | None = None,
    best_objective: float | None = None,
    objective: str | None = None,
    detail: str | None = None,
) -> None:
    """Report progress to the installed reporter, if any.

    ``detail`` is added to the message, after the count and best objective.
    """
    reporter = _REPORTER.get()
    if reporter is None:
        return
    reporter.emit(
        analysis=analysis,
        stage=stage,
        done=done,
        total=total,
        unit=unit,
        best_objective=best_objective,
        objective=objective,
        detail=detail,
    )


def _message(
    *,
    analysis: str,
    stage: str,
    done: int | None,
    total: int | None,
    unit: str | None,
    best_objective: float | None,
    objective: str | None,
    detail: str | None,
) -> str:
    parts = []
    if done is not None:
        count = f"{done} of {total}" if total is not None else f"{done}"
        parts.append(f"{count} {unit}" if unit else count)
    if best_objective is not None and math.isfinite(best_objective):
        parts.append(f"best {objective or 'objective'} {best_objective:.4g}")
    if detail:
        parts.append(detail)
    title = stage.replace("_", " ").capitalize()
    return f"{title}: {', '.join(parts)}" if parts else title
