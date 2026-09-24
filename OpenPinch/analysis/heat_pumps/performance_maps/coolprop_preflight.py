"""Detached CoolProp capability preflight for optimized HPR topologies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ....contracts.hpr import (
    HeatPumpTargetInputs,
    HPRFailureCategory,
    HPRFailureDiagnostic,
    HPRFailureSummary,
    HPRParsedState,
    HPRSearchBudget,
    HPRTargetingError,
    HPRTopologyIdentifier,
)
from ....domain.fluids import build_coolprop_abstract_state
from .fluids import parse_hpr_working_fluid, resolve_hpr_working_fluid


@dataclass(frozen=True, slots=True)
class PreparedCoolPropStageCapability:
    """One engine-free stage capability proven before HPR search."""

    ordinal: int
    role: Literal["vc", "mvr"]
    fluid_spec: str
    evaporating_or_suction_temperature: float
    condensing_or_discharge_temperature: float
    representative_state_supported: bool = True


def preflight_coolprop_fluid_names(args: HeatPumpTargetInputs, topology_id) -> None:
    """Reject unsupported identities before even the Carnot initialization search."""
    topology = HPRTopologyIdentifier(topology_id)
    vc_fluids = args.refrigerant_ls or ["Water"]
    vc_fluids = [vc_fluids] if isinstance(vc_fluids, str) else list(vc_fluids)
    roles = ["vc"] * len(vc_fluids)
    fluids = vc_fluids
    if topology is HPRTopologyIdentifier.VAPOUR_COMPRESSION_MVR:
        mvr_fluids = getattr(args, "mvr_fluid_ls", None) or ["Water"]
        mvr_fluids = [mvr_fluids] if isinstance(mvr_fluids, str) else mvr_fluids
        roles += ["mvr"] * len(mvr_fluids)
        fluids += list(mvr_fluids)
    for ordinal, (role, fluid) in enumerate(zip(roles, fluids, strict=True)):
        try:
            parsed = parse_hpr_working_fluid(fluid)
            if parsed.property_backend == "REFPROP":
                raise ValueError("REFPROP is not supported")
            build_coolprop_abstract_state(parsed.source_spec)
        except ValueError as exc:
            diagnostic = HPRFailureDiagnostic(
                category=HPRFailureCategory.PREFLIGHT_REJECTION,
                reason_code="coolprop.preflight.unsupported_fluid",
                summary=f"CoolProp rejected {role} stage {ordinal} fluid identity",
                fluid=" ".join(str(fluid).split())[:256] or "<empty>",
                stage_index=ordinal,
                topology=topology,
            )
            raise HPRTargetingError(
                diagnostic.summary,
                diagnostics=HPRFailureSummary(
                    simulation_backend="coolprop",
                    cycle=topology.value,
                    evaluated_count=0,
                    category_counts={HPRFailureCategory.PREFLIGHT_REJECTION: 1},
                    representative_failures=(diagnostic,),
                    budget=getattr(args, "search_budget", None) or HPRSearchBudget(),
                    warm_start_evaluated=False,
                    warm_start_viable=False,
                ),
            ) from exc


@dataclass(frozen=True, slots=True)
class PreparedCoolPropTargeting:
    """Ordered CoolProp stage capabilities for one optimization request."""

    topology_id: HPRTopologyIdentifier
    stages: tuple[PreparedCoolPropStageCapability, ...]


def preflight_coolprop_hpr_targeting(
    *,
    args: HeatPumpTargetInputs,
    state: HPRParsedState,
    topology_id: HPRTopologyIdentifier | str,
) -> PreparedCoolPropTargeting:
    """Check identities and probe seeds without rejecting the entire search space."""
    topology = HPRTopologyIdentifier(topology_id)
    preflight_coolprop_fluid_names(args, topology)
    T_evap = _finite_vector(state.T_evap, name="T_evap")
    T_cond = _finite_vector(state.T_cond, name="T_cond")
    if T_evap.size != T_cond.size or not T_evap.size:
        raise ValueError("CoolProp preflight requires aligned nonempty temperatures.")

    roles = _stage_roles(topology, args=args, count=T_evap.size)
    fluids = _stage_fluids(roles, args=args)
    stages: list[PreparedCoolPropStageCapability] = []
    for ordinal, (role, fluid, T_source, T_sink) in enumerate(
        zip(roles, fluids, T_evap, T_cond, strict=True)
    ):
        supported = True
        try:
            resolve_hpr_working_fluid(
                fluid,
                evaporating_temperature=float(T_source),
                condensing_temperature=float(T_sink),
            )
        except ValueError:
            # An unsupported seed is candidate-local, not proof that no fluid
            # state inside the variable bounds can work.
            supported = False
        stages.append(
            PreparedCoolPropStageCapability(
                ordinal=ordinal,
                role=role,
                fluid_spec=fluid,
                evaporating_or_suction_temperature=float(T_source),
                condensing_or_discharge_temperature=float(T_sink),
                representative_state_supported=supported,
            )
        )
    return PreparedCoolPropTargeting(topology_id=topology, stages=tuple(stages))


def representative_hpr_point(
    initial_points: object,
    bounds: object,
) -> np.ndarray:
    """Return the first warm start or the deterministic bounds midpoint."""
    if initial_points is None:
        return np.asarray(bounds, dtype=float).mean(axis=1)
    values = np.asarray(initial_points, dtype=float)
    if values.size:
        return values if values.ndim == 1 else values.reshape(values.shape[0], -1)[0]
    bound_array = np.asarray(bounds, dtype=float)
    return bound_array.mean(axis=1)


def _stage_roles(
    topology: HPRTopologyIdentifier,
    *,
    args: HeatPumpTargetInputs,
    count: int,
) -> tuple[Literal["vc", "mvr"], ...]:
    if topology is not HPRTopologyIdentifier.VAPOUR_COMPRESSION_MVR:
        return ("vc",) * count
    n_mvr = int(args.n_mvr)
    if n_mvr < 1 or n_mvr >= count:
        raise ValueError("VC+MVR preflight requires VC and MVR stages.")
    return ("mvr",) * n_mvr + ("vc",) * (count - n_mvr)


def _stage_fluids(
    roles: tuple[Literal["vc", "mvr"], ...],
    *,
    args: HeatPumpTargetInputs,
) -> tuple[str, ...]:
    vc_fluids = _normalize_fluids(args.refrigerant_ls, roles.count("vc"), "VC")
    mvr_fluids = _normalize_fluids(
        getattr(args, "mvr_fluid_ls", []),
        roles.count("mvr"),
        "MVR",
    )
    vc_index = 0
    mvr_index = 0
    result: list[str] = []
    for role in roles:
        if role == "mvr":
            result.append(mvr_fluids[mvr_index])
            mvr_index += 1
        else:
            result.append(vc_fluids[vc_index])
            vc_index += 1
    return tuple(result)


def _normalize_fluids(values: object, count: int, label: str) -> tuple[str, ...]:
    if count == 0:
        return ()
    fluids = [str(value).strip() for value in values]
    if not fluids or any(not fluid for fluid in fluids):
        raise ValueError(f"{label} preflight requires nonempty fluid names.")
    return tuple((fluids + [fluids[-1]] * count)[:count])


def _finite_vector(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


__all__ = [
    "PreparedCoolPropStageCapability",
    "PreparedCoolPropTargeting",
    "preflight_coolprop_hpr_targeting",
    "representative_hpr_point",
]
