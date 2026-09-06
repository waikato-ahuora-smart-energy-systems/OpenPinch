"""Build application-owned contexts and validate references to prior analyses."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any

from ....domain.analysis import AnalysisProvenance
from .execution import walk_zone_tree


def _json_default(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"Cannot fingerprint {type(value).__name__}")


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, default=_json_default, allow_nan=False)


def input_fingerprint(problem) -> str:
    root = problem._master_zone
    payload = {
        "inputs": problem._problem_data,
        "project_name": problem.project_name,
        "zones": []
        if root is None
        else [(z.address, z.type, z.dt_cont_multiplier) for z in walk_zone_tree(root)],
        "components": [
            (
                key,
                c.component_type,
                c.active,
                getattr(c, "settings", None),
                [
                    (
                        r.original_stream.name,
                        [(m.zone.address, m.key) for m in r.original_memberships],
                    )
                    for r in getattr(c, "stream_records", ())
                ],
            )
            for key, c in sorted(problem._process_components.items())
        ],
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def make_provenance(
    problem, method_id, zone, *, period_ids, settings=None, prerequisites=()
) -> AnalysisProvenance:
    return AnalysisProvenance(
        owner_id=problem._analysis_owner_id,
        method_id=method_id,
        zone_address=zone.address,
        period_ids=tuple(period_ids),
        input_fingerprint=input_fingerprint(problem),
        effective_settings_json=canonical_json(settings or zone.config._values),
        prerequisite_ids=tuple(prerequisites),
    )


def validate_base_target(problem, target, *, zone, period_id, options=None):
    """Reject foreign, stale, and mismatched references before any calculation."""
    selected = problem._resolve_target_zone(zone)
    runtime, sid = problem._resolve_runtime_period_options(
        {**dict(options or {}), **({"period_id": period_id} if period_id else {})},
        zone=selected,
    )
    provenance = getattr(target, "provenance", None)
    current = selected.targets.get(getattr(target, "type", None))
    if (
        provenance is None
        or provenance.owner_id != problem._analysis_owner_id
        or provenance.input_fingerprint != input_fingerprint(problem)
        or provenance.zone_address != selected.address
        or getattr(target, "period_idx", None) != runtime["period_idx"]
        or current is None
        or current.provenance != provenance
    ):
        raise ValueError(
            "base_target must reference a current local target for this zone and period"
        )
    return target.type


def stamp_targets(problem, surface: str, previous: dict[int, Any]) -> None:
    """Stamp newly created/enriched targets with their effective calculation inputs."""
    # Children and same-zone prerequisites precede their dependent results.
    old_by_key = {(t.scope, t.type, t.period_idx): t for t in previous.values()}
    for zone in reversed(list(walk_zone_tree(problem._master_zone))):
        pending = [t for t in zone.targets.values() if id(t) not in previous]
        while pending:
            ready = next(
                (
                    t
                    for t in pending
                    if not any(
                        dependency.type in t.prerequisite_types()
                        for dependency in pending
                    )
                ),
                pending[0],
            )
            pending = [target for target in pending if target is not ready]
            sid = ready.period_id
            if sid is None and ready.period_idx is not None:
                sid = next(
                    (s for s, i in zone.period_ids.items() if i == ready.period_idx),
                    None,
                )
            candidates = [
                t
                for t in zone.targets.values()
                if t.type in ready.prerequisite_types() and t is not ready
            ]
            if ready.prerequisite_types():
                candidates.extend(
                    t
                    for child in zone.subzones.values()
                    for t in child.targets.values()
                    if t.type in ready.prerequisite_types()
                )
            if surface in {
                "exergy",
                "cogeneration",
                "sun_smith_cogeneration",
                "varbanov_cogeneration",
                "isentropic_cogeneration",
                "heat_exchanger_area_and_cost",
            }:
                old = old_by_key.get((ready.scope, ready.type, ready.period_idx))
                if old is not None:
                    candidates.append(old)
            prerequisites = tuple(
                dict.fromkeys(
                    t.provenance.identity
                    for t in candidates
                    if t.provenance is not None and t.period_idx == ready.period_idx
                )
            )
            ready.provenance = make_provenance(
                problem,
                "target." + surface,
                zone,
                period_ids=() if sid is None else (sid,),
                settings=ready.config._values,
                prerequisites=prerequisites,
            )
