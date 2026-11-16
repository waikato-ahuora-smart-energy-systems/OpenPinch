"""The heat pump and refrigeration cycles, each described once.

A cycle spec names the model family (``HeatPumpAndRefrigerationCycle``) each
of its topologies selects, the duties it serves, its own options and the
``HPR_*`` configuration keys they set, its optional dependencies, the
screening methods a fluid-set run uses for each topology, and whether it is
available and runs across all periods. Adding a cycle means writing its
model and registering a spec here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

from pydantic import BaseModel, ValidationError

from ....contracts.hpr import (
    BraytonCycleOptions,
    CarnotCycleOptions,
    HPRTopologyIdentifier,
    MvrCycleOptions,
    VapourCompressionCycleOptions,
)
from ....domain.configuration_fields import register_hpr_type
from ....domain.enums import HeatPumpAndRefrigerationCycle

DUTIES = frozenset({"heat_pump", "refrigeration"})


@dataclass(frozen=True)
class CycleFamilyHandlers:
    """The analysis code one model family runs, as lazy ``module:function`` paths.

    ``placement`` optimises one period's heat pump placement;
    ``multiperiod_setup`` returns the shared multiperiod starts, bounds,
    objective and stage lifts (``None``: no multiperiod support);
    ``topology_identifier`` names the simulated topology, for CoolProp
    preflight and the accepted target (``None`` for idealised cycles).
    """

    placement: str
    multiperiod_setup: str | None = None
    topology_identifier: HPRTopologyIdentifier | None = None


def resolve_handler(path: str | None):
    """The function a ``package.module:function`` path names, imported now."""

    if path is None:
        return None
    module, name = path.split(":")
    return getattr(import_module(module), name)


@dataclass(frozen=True)
class HprCycleSpec:
    """One registered heat pump and refrigeration cycle."""

    id: str
    label: str
    # A topology's model family: a HeatPumpAndRefrigerationCycle member, or
    # the HPR_TYPE string of a family registered with the cycle.
    topologies: Mapping[str, HeatPumpAndRefrigerationCycle | str]
    default_topology: str
    duties: frozenset[str]
    options_model: type[BaseModel]
    # Option name -> HPR_* configuration key it sets when given.
    configuration_keys: Mapping[str, str] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    screening_methods: Mapping[str, tuple[HPRTopologyIdentifier, ...]] = field(
        default_factory=dict
    )
    available: bool = True
    all_periods: bool = True
    # Topology -> the analysis handlers of the model family it selects.
    handlers: Mapping[str, CycleFamilyHandlers] = field(default_factory=dict)

    def to_configuration(self, options) -> dict[str, Any]:
        """The ``HPR_*`` configuration the given options set.

        ``options`` is this cycle's options model, a mapping validated by it,
        or ``None``; options left unset set nothing.
        """

        if options is None:
            return {}
        if not isinstance(options, self.options_model):
            try:
                options = self.options_model.model_validate(dict(options))
            except ValidationError as error:
                raise ValueError(
                    f"invalid options for the {self.id} cycle: {error}"
                ) from error
        return {
            key: getattr(options, name)
            for name, key in self.configuration_keys.items()
            if getattr(options, name) is not None
        }


_REGISTRY: dict[str, HprCycleSpec] = {}


def register_cycle(spec: HprCycleSpec) -> HprCycleSpec:
    """Add ``spec``; its id and model families must be new."""

    if spec.id in _REGISTRY:
        raise ValueError(f"cycle {spec.id!r} is already registered.")
    taken = {
        family_value(member)
        for other in _REGISTRY.values()
        for member in other.topologies.values()
    }
    if taken & {family_value(member) for member in spec.topologies.values()}:
        raise ValueError(f"cycle {spec.id!r} reuses another cycle's model family.")
    if spec.default_topology not in spec.topologies:
        raise ValueError(
            f"cycle {spec.id!r} has no {spec.default_topology!r} topology."
        )
    if set(spec.handlers) != set(spec.topologies):
        raise ValueError(f"cycle {spec.id!r} needs handlers for every topology.")
    if not spec.duties or not spec.duties <= DUTIES:
        raise ValueError(f"cycle {spec.id!r} duties must be among {sorted(DUTIES)}.")
    for member in spec.topologies.values():
        if isinstance(member, str):
            # A family of the cycle's own becomes a valid HPR_TYPE.
            register_hpr_type(member)
    _REGISTRY[spec.id] = spec
    return spec


def cycle_spec(cycle_id: str) -> HprCycleSpec:
    """The registered spec with id ``cycle_id``."""

    try:
        return _REGISTRY[cycle_id]
    except KeyError:
        raise ValueError(
            f"unknown cycle {cycle_id!r}; registered cycles are "
            + ", ".join(repr(name) for name in _REGISTRY)
            + "."
        ) from None


def registered_cycles() -> tuple[HprCycleSpec, ...]:
    """Every registered spec, in registration order."""

    return tuple(_REGISTRY.values())


def family_value(member: HeatPumpAndRefrigerationCycle | str) -> str:
    """A model family's ``HPR_TYPE`` value."""

    return member.value if isinstance(member, HeatPumpAndRefrigerationCycle) else member


def spec_for_cycle(
    member: HeatPumpAndRefrigerationCycle | str,
) -> tuple[HprCycleSpec, str]:
    """The spec and topology that select the model family ``member``."""

    value = family_value(member)
    for spec in _REGISTRY.values():
        for topology, family in spec.topologies.items():
            if family_value(family) == value:
                return spec, topology
    raise ValueError(f"no registered cycle selects {value!r}.")


def family_handlers(hpr_type: str) -> CycleFamilyHandlers:
    """The handlers of the model family whose ``HPR_TYPE`` value is ``hpr_type``."""

    for spec in _REGISTRY.values():
        for topology, family in spec.topologies.items():
            if family_value(family) == hpr_type:
                return spec.handlers[topology]
    raise ValueError(f"no registered cycle selects {hpr_type!r}.")


_TARGETING = "OpenPinch.analysis.heat_pumps.targeting"
_SETUP = "OpenPinch.analysis.heat_pumps._multiperiod.setup"

_EFFICIENCY_KEYS = {
    "compressor_efficiency": "HPR_ETA_COMP",
    "expander_efficiency": "HPR_ETA_EXP",
}
_VAPOUR_COMPRESSION_KEYS = _EFFICIENCY_KEYS | {
    "refrigerants": "HPR_REFRIGERANTS",
    "initialize_from_carnot": "HPR_INITIALISE_SIMULATED_CYCLE",
    "sort_refrigerants": "HPR_REFRIGERANT_SORT_ENABLED",
    "allow_integrated_expander": "HPR_INTEGRATED_EXPANDER_ENABLED",
    "compressor_model": "HPR_COMPRESSOR_MODEL",
}

register_cycle(
    HprCycleSpec(
        id="carnot",
        label="Carnot cycles",
        topologies={
            "cascade": HeatPumpAndRefrigerationCycle.CascadeCarnot,
            "parallel": HeatPumpAndRefrigerationCycle.ParallelCarnot,
        },
        default_topology="cascade",
        duties=DUTIES,
        options_model=CarnotCycleOptions,
        configuration_keys=_EFFICIENCY_KEYS,
        handlers={
            "cascade": CycleFamilyHandlers(
                placement=f"{_TARGETING}.cascade_carnot"
                ":optimise_cascade_carnot_heat_pump_placement",
                multiperiod_setup=f"{_SETUP}:_cascade_carnot_setup",
            ),
            "parallel": CycleFamilyHandlers(
                placement=f"{_TARGETING}.parallel_carnot"
                ":optimise_parallel_carnot_heat_pump_placement",
                multiperiod_setup=f"{_SETUP}:_parallel_carnot_setup",
            ),
        },
    )
)
register_cycle(
    HprCycleSpec(
        id="vapour_compression",
        label="Vapour compression cycles",
        topologies={
            "cascade": HeatPumpAndRefrigerationCycle.CascadeVapourComp,
            "parallel": HeatPumpAndRefrigerationCycle.ParallelVapourComp,
        },
        default_topology="cascade",
        duties=DUTIES,
        options_model=VapourCompressionCycleOptions,
        configuration_keys=_VAPOUR_COMPRESSION_KEYS,
        dependencies=("tespy (explicit backend only)",),
        screening_methods={
            "cascade": (
                HPRTopologyIdentifier.SINGLE_STAGE_VAPOUR_COMPRESSION,
                HPRTopologyIdentifier.CASCADE_VAPOUR_COMPRESSION,
            ),
            "parallel": (
                HPRTopologyIdentifier.SINGLE_STAGE_VAPOUR_COMPRESSION,
                HPRTopologyIdentifier.PARALLEL_VAPOUR_COMPRESSION,
            ),
        },
        handlers={
            "cascade": CycleFamilyHandlers(
                placement=f"{_TARGETING}.cascade_vapour_compression"
                ":optimise_cascade_heat_pump_placement",
                multiperiod_setup=f"{_SETUP}:_cascade_vapour_compression_setup",
                topology_identifier=HPRTopologyIdentifier.CASCADE_VAPOUR_COMPRESSION,
            ),
            "parallel": CycleFamilyHandlers(
                placement=f"{_TARGETING}.parallel_vapour_compression"
                ":optimise_parallel_heat_pump_placement",
                multiperiod_setup=f"{_SETUP}:_parallel_vapour_compression_setup",
                topology_identifier=HPRTopologyIdentifier.PARALLEL_VAPOUR_COMPRESSION,
            ),
        },
    )
)
register_cycle(
    HprCycleSpec(
        id="mvr",
        label="Vapour compression with MVR cascade",
        topologies={"cascade": HeatPumpAndRefrigerationCycle.VapourCompMVR},
        default_topology="cascade",
        duties=frozenset({"heat_pump"}),
        options_model=MvrCycleOptions,
        configuration_keys={
            "refrigerants": "HPR_REFRIGERANTS",
            "mvr_fluids": "HPR_MVR_FLUIDS",
            "mvr_compressor_efficiency": "HPR_MVR_ETA_COMP",
            "mvr_stages": "HPR_MVR_COUNT",
            "motor_efficiency": "HPR_MVR_ETA_MOTOR",
            "compressor_model": "HPR_COMPRESSOR_MODEL",
            "mvr_compressor_model": "HPR_MVR_COMPRESSOR_MODEL",
        },
        screening_methods={"cascade": (HPRTopologyIdentifier.VAPOUR_COMPRESSION_MVR,)},
        handlers={
            "cascade": CycleFamilyHandlers(
                placement=f"{_TARGETING}.vapour_compression_mvr"
                ":optimise_vapour_compression_mvr_heat_pump_placement",
                multiperiod_setup=f"{_SETUP}:_vapour_compression_mvr_setup",
                topology_identifier=HPRTopologyIdentifier.VAPOUR_COMPRESSION_MVR,
            )
        },
    )
)
register_cycle(
    HprCycleSpec(
        id="brayton",
        label="Brayton cycle",
        topologies={"single": HeatPumpAndRefrigerationCycle.Brayton},
        default_topology="single",
        duties=DUTIES,
        options_model=BraytonCycleOptions,
        configuration_keys=_EFFICIENCY_KEYS,
        available=False,
        all_periods=False,
        handlers={
            "single": CycleFamilyHandlers(
                placement=f"{_TARGETING}.brayton:optimise_brayton_heat_pump_placement"
            )
        },
    )
)


__all__ = [
    "DUTIES",
    "CycleFamilyHandlers",
    "HprCycleSpec",
    "cycle_spec",
    "family_handlers",
    "family_value",
    "register_cycle",
    "registered_cycles",
    "resolve_handler",
    "spec_for_cycle",
]
