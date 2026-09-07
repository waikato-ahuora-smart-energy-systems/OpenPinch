"""Read-only validation of locally produced HPR target handles."""

import hashlib

from ..domain.enums import TargetType
from ..domain.targets import HeatPumpTargetBase
from ._problem.targeting.provenance import canonical_json, input_fingerprint


def hpr_result_digest(target):
    def stream_signature(stream):
        idx = target.period_idx
        return (
            stream.name,
            stream.is_active,
            stream.fluid_name,
            stream.fluid_phase,
            [
                (
                    attribute,
                    None
                    if getattr(stream, attribute) is None
                    else (
                        str(getattr(stream, attribute).unit),
                        float(getattr(stream, attribute)[idx]).hex(),
                    ),
                )
                for attribute in (
                    "supply_temperature",
                    "target_temperature",
                    "heat_flow",
                    "price",
                    "effective_delta_t_contribution",
                    "maximum_heat_flow",
                    "heat_transfer_coefficient",
                    "supply_pressure",
                    "target_pressure",
                    "supply_enthalpy",
                    "target_enthalpy",
                )
            ],
            [stream_signature(part) for part in stream.segments],
            stream._segment_targeting_fractions.get(idx),
        )

    payload = {
        "load": target.hpr_load,
        "settings": target.provenance_settings(),
        "utilities": [
            stream_signature(u) for u in target.hot_utilities + target.cold_utilities
        ],
        "cycle_streams": [
            stream_signature(u)
            for u in target.hpr_hot_streams + target.hpr_cold_streams
        ],
        "cycle_result": {
            field: getattr(target.hpr_details, field, None)
            for field in (
                "w_net",
                "T_cond",
                "T_evap",
                "Q_cond",
                "Q_evap",
                "Q_amb_hot",
                "Q_amb_cold",
            )
        },
        "residual": target.hpr_residual,
        "type": target.type,
        "scope": target.scope,
        "period": target.period_id,
        "period_idx": target.period_idx,
        "zone_name": target.zone_name,
        "zone_type": target.zone_type,
        "graphs": {
            key: (table.columns, hashlib.sha256(table.data.tobytes()).hexdigest())
            for key, table in target.graphs.items()
        },
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def retain_hpr_target(problem, target):
    if isinstance(target, HeatPumpTargetBase) and target.provenance is not None:
        registry = dict(getattr(problem, "_retained_hpr_targets", {}))
        registry[target.provenance.identity] = hpr_result_digest(target)
        problem._retained_hpr_targets = registry


def validate_hpr_target(problem, target):
    if not isinstance(target, HeatPumpTargetBase) or target.provenance is None:
        raise ValueError("Select a solved local HPR target.")
    provenance = target.provenance
    if (
        provenance.owner_id != problem._analysis_owner_id
        or provenance.input_fingerprint != input_fingerprint(problem)
        or getattr(problem, "_retained_hpr_targets", {}).get(provenance.identity)
        != hpr_result_digest(target)
    ):
        raise ValueError("HPR target is foreign, stale or modified; solve it again.")
    return target


def select_hpr_graphs(problem, *, target, mode, zone_name=None):
    from ..analysis.graphs.service import _create_graph_set

    if target is None and problem._master_zone is None:
        raise ValueError("No solved HPR target is available.")
    if target is None:
        candidates = [
            t
            for z in problem._walk_zone_tree(problem._master_zone)
            for t in z.targets.values()
            if isinstance(t, HeatPumpTargetBase)
            and (
                (t.type in (TargetType.DHP.value, TargetType.IHP.value))
                == (mode == "heat_pump")
            )
            and (zone_name is None or zone_name in (t.scope, t.zone_name, t.name))
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Select target= explicitly; no unique matching HPR target is available."
            )
        target = candidates[0]
    validate_hpr_target(problem, target)
    heating = target.type in (TargetType.DHP.value, TargetType.IHP.value)
    if heating != (mode == "heat_pump") or (
        zone_name is not None
        and zone_name not in (target.scope, target.zone_name, target.name)
    ):
        raise ValueError("HPR target does not match the requested mode or zone.")
    return _create_graph_set(target)["graphs"]
