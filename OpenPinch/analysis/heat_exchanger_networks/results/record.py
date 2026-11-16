"""The live record of verified networks one HEN synthesis run admits.

Every network a run produces and solves is offered to :func:`admit`. A network
that passes the admission rule enters each record installed with
:func:`recording`, in ascending total annual cost, one entry per topology. The
record is thread-safe, so another thread can poll it while the run works.
"""

from __future__ import annotations

import contextvars
import copy
import hashlib
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from ....contracts.synthesis.task import HeatExchangerNetworkSynthesisTaskOutcome
from ..reporting.ranking import network_structure_signature
from ..reporting.verification import _count_active


@dataclass(frozen=True)
class _Entry:
    key: str
    sort_key: tuple[float, str, str]
    outcome: HeatExchangerNetworkSynthesisTaskOutcome
    version: int
    data: dict


class HeatExchangerNetworkRecord:
    """Verified networks of one synthesis run, cheapest first."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._version = 0
        self._status = "pending"
        self._selected: str | None = None
        self._entries: dict[str, _Entry] = {}

    @property
    def version(self) -> int:
        """Rises by one on every change to the record."""
        with self._lock:
            return self._version

    @property
    def status(self) -> str:
        """``"pending"``, ``"running"``, ``"complete"`` or ``"failed"``."""
        with self._lock:
            return self._status

    def outcomes(self) -> tuple[HeatExchangerNetworkSynthesisTaskOutcome, ...]:
        """The admitted outcomes, cheapest first."""
        with self._lock:
            return tuple(entry.outcome for entry in self._ordered())

    def snapshot(self) -> dict:
        """The record as JSON-ready data, built afresh on every call."""
        with self._lock:
            return self._header() | {"entries": self._entry_data(0)}

    def changed_since(self, version: int) -> dict | None:
        """What changed after ``version``, or ``None`` if nothing did.

        ``entries`` holds only the entries added or replaced after
        ``version``; ``order`` lists every entry's key in rank order. Entries
        are never removed, so a reader merges the changed entries by key.
        """
        with self._lock:
            if version >= self._version:
                return None
            return self._header() | {
                "entries": self._entry_data(version),
                "order": [entry.key for entry in self._ordered()],
            }

    def _header(self) -> dict:
        return {
            "version": self._version,
            "status": self._status,
            "selected": self._selected,
        }

    def _set_status(self, status: str, *, expected: str | None = None) -> None:
        with self._lock:
            if expected is not None and self._status != expected:
                raise ValueError(
                    "a HeatExchangerNetworkRecord collects one run; pass a new "
                    "record to each design call."
                )
            if status == "complete":
                ordered = self._ordered()
                self._selected = ordered[0].key if ordered else None
            self._status = status
            self._version += 1

    def _entry_data(self, after_version: int) -> list[dict]:
        return [
            copy.deepcopy(entry.data) | {"rank": rank}
            for rank, entry in enumerate(self._ordered(), start=1)
            if entry.version > after_version
        ]

    def _ordered(self) -> list[_Entry]:
        return sorted(self._entries.values(), key=lambda entry: entry.sort_key)

    def _insert(self, outcome: HeatExchangerNetworkSynthesisTaskOutcome) -> bool:
        signature = network_structure_signature(outcome.network)
        key = hashlib.sha1(repr(signature).encode()).hexdigest()[:16]
        sort_key = (
            float(outcome.network.total_annual_cost),
            outcome.task.task_id or "",
            key,
        )
        data = _entry_json(key, outcome)
        with self._lock:
            current = self._entries.get(key)
            if current is not None and current.sort_key <= sort_key:
                return False
            self._version += 1
            self._entries[key] = _Entry(
                key, sort_key, outcome, self._version, data | {"version": self._version}
            )
            return True


def _entry_json(key: str, outcome: HeatExchangerNetworkSynthesisTaskOutcome) -> dict:
    network = outcome.network
    return {
        "key": key,
        "method": str(outcome.task.method),
        "task_id": outcome.task.task_id,
        "dt_cont_multiplier": float(outcome.task.dt_cont_multiplier),
        "total_annual_cost": float(network.total_annual_cost),
        "unit_count": _count_active(network.exchangers, None),
        "network": network.model_dump(mode="json"),
    }


Finalise = Callable[
    [HeatExchangerNetworkSynthesisTaskOutcome],
    HeatExchangerNetworkSynthesisTaskOutcome | None,
]


@dataclass(frozen=True)
class _Destination:
    record: HeatExchangerNetworkRecord
    finalise: Finalise | None


_DESTINATIONS: ContextVar[tuple[_Destination, ...]] = ContextVar(
    "openpinch_hen_network_records", default=()
)
# The record whose run builds the result; set by ``running``.
_RUN: ContextVar[HeatExchangerNetworkRecord | None] = ContextVar(
    "openpinch_hen_run_record", default=None
)


@contextmanager
def recording(
    record: HeatExchangerNetworkRecord, *, finalise: Finalise | None = None
) -> Iterator[None]:
    """Send the networks the enclosed run admits to ``record`` as well.

    ``finalise`` turns an admitted network into its final form (for example
    fitted to every period and rated) before it enters ``record``; it returns
    ``None`` when it cannot, and its result must pass the admission rule too.
    """
    token = _DESTINATIONS.set(_DESTINATIONS.get() + (_Destination(record, finalise),))
    try:
        yield
    finally:
        _DESTINATIONS.reset(token)


@contextmanager
def running(
    record: HeatExchangerNetworkRecord, *, finalise: Finalise | None = None
) -> Iterator[None]:
    """Run the enclosed synthesis into ``record`` and settle its status.

    The record is ``"running"`` inside, ``"complete"`` with its first entry
    selected when the block ends, and ``"failed"`` when the block raises.
    ``finalise`` is as for :func:`recording`.
    """
    record._set_status("running", expected="pending")
    token = _RUN.set(record)
    try:
        with recording(record, finalise=finalise):
            yield
    except BaseException:
        record._set_status("failed")
        raise
    finally:
        _RUN.reset(token)
    record._set_status("complete")


def run_record(
    outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
) -> HeatExchangerNetworkRecord:
    """The record of the enclosing run, or one holding ``outcomes`` alone.

    Outside :func:`running`, for example when a workflow is called directly,
    the admissible ``outcomes`` make up the record.
    """
    record = _RUN.get()
    if record is not None:
        return record
    record = HeatExchangerNetworkRecord()
    for outcome in outcomes:
        if _admissible(outcome):
            record._insert(outcome)
    return record


def admit(outcome: HeatExchangerNetworkSynthesisTaskOutcome) -> bool:
    """Offer one solved outcome to the installed records.

    Returns whether any record changed. An outcome is admitted only when it
    succeeded with a network and a total annual cost, converged, and passes
    the post-solve feasibility checks of the final ranking.
    """
    destinations = _DESTINATIONS.get()
    if not destinations or not _admissible(outcome):
        return False
    changed = False
    for destination in destinations:
        final = outcome
        if destination.finalise is not None:
            # A fit solves too; what it admits must not enter this run.
            final = contextvars.copy_context().run(
                _outside_records, destination, outcome
            )
            if final is None or not _admissible(final):
                continue
        changed = destination.record._insert(final) or changed
    return changed


def _outside_records(
    destination: _Destination, outcome: HeatExchangerNetworkSynthesisTaskOutcome
) -> HeatExchangerNetworkSynthesisTaskOutcome | None:
    _DESTINATIONS.set(())
    _RUN.set(None)
    return destination.finalise(outcome)


def _admissible(outcome: HeatExchangerNetworkSynthesisTaskOutcome) -> bool:
    from .assembly import demote_infeasible_outcomes

    if (
        outcome.status != "success"
        or outcome.network is None
        or outcome.network.total_annual_cost is None
    ):
        return False
    (checked,) = demote_infeasible_outcomes((outcome,))
    return checked.status == "success"


__all__ = [
    "HeatExchangerNetworkRecord",
    "admit",
    "recording",
    "run_record",
    "running",
]
