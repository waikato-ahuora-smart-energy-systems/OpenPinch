"""Compressor-work accounting for Process MVR components."""

from __future__ import annotations

from ....domain.zone import Zone
from ..direct_mvr.models import DirectGasMVRStageResult
from .membership import record_affects_zone
from .state import _ProcessMVRStreamRecord


def work_for_zone(
    records: list[_ProcessMVRStreamRecord],
    *,
    active: bool,
    zone: Zone,
    period_id: str | None,
    period_idx: int | None,
) -> float:
    """Return compressor work for records in a zone subtree and period."""
    if not active:
        return 0.0
    total = 0.0
    for record in records:
        if not record_affects_zone(record, zone):
            continue
        total += sum(
            stage.work
            for stage in record_stage_results_for_period(
                record,
                period_id=period_id,
                period_idx=period_idx,
            )
        )
    return float(total)


def record_stage_results_for_period(
    record: _ProcessMVRStreamRecord,
    *,
    period_id: str | None,
    period_idx: int | None,
) -> list[DirectGasMVRStageResult]:
    """Resolve one record's stage results from period identity or index."""
    if period_id is not None and period_id in record.stage_results_by_period:
        return record.stage_results_by_period[period_id]
    if period_idx is not None:
        index_label = record.period_label_by_index.get(
            int(period_idx),
            str(period_idx),
        )
        if index_label in record.stage_results_by_period:
            return record.stage_results_by_period[index_label]
    return next(iter(record.stage_results_by_period.values()), [])


__all__: list[str] = []
