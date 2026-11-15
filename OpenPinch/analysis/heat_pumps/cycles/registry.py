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
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from ....contracts.hpr import HPRTopologyIdentifier
from ....domain.enums import HeatPumpAndRefrigerationCycle

DUTIES = frozenset({"heat_pump", "refrigeration"})


class _CycleOptions(BaseModel):
    """A cycle's own options. Values pass unchanged to the ``HPR_*``
    configuration, whose validators check them; only unknown names are
    rejected here."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class CarnotOptions(_CycleOptions):
    """Carnot cycles have no options of their own."""


class BraytonOptions(_CycleOptions):
    """Brayton cycles have no options of their own."""


class VapourCompressionOptions(_CycleOptions):
    """Refrigerants and simulation choices of vapour compression cycles.

    ``compressor_model`` may be a compressor efficiency function in a call;
    such a function is not stored with the case.
    """

    refrigerants: Any = None
    initialize_from_carnot: Any = None
    sort_refrigerants: Any = None
    allow_integrated_expander: Any = None
    compressor_model: Any = None


class MvrOptions(_CycleOptions):
    """Refrigerants, MVR fluids and stages, and efficiencies of an MVR cascade."""

    refrigerants: Any = None
    mvr_fluids: Any = None
    mvr_compressor_efficiency: Any = None
    mvr_compressor_model: Any = None
    compressor_model: Any = None
    mvr_stages: Any = None
    motor_efficiency: Any = None


@dataclass(frozen=True)
class HprCycleSpec:
    """One registered heat pump and refrigeration cycle."""

    id: str
    label: str
    topologies: Mapping[str, HeatPumpAndRefrigerationCycle]
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
        member for other in _REGISTRY.values() for member in other.topologies.values()
    }
    if taken & set(spec.topologies.values()):
        raise ValueError(f"cycle {spec.id!r} reuses another cycle's model family.")
    if spec.default_topology not in spec.topologies:
        raise ValueError(
            f"cycle {spec.id!r} has no {spec.default_topology!r} topology."
        )
    if not spec.duties or not spec.duties <= DUTIES:
        raise ValueError(f"cycle {spec.id!r} duties must be among {sorted(DUTIES)}.")
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


def spec_for_cycle(member: HeatPumpAndRefrigerationCycle) -> tuple[HprCycleSpec, str]:
    """The spec and topology that select the model family ``member``."""

    for spec in _REGISTRY.values():
        for topology, family in spec.topologies.items():
            if family is member:
                return spec, topology
    raise ValueError(f"no registered cycle selects {member.value!r}.")


_VAPOUR_COMPRESSION_KEYS = {
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
        options_model=CarnotOptions,
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
        options_model=VapourCompressionOptions,
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
    )
)
register_cycle(
    HprCycleSpec(
        id="mvr",
        label="Vapour compression with MVR cascade",
        topologies={"cascade": HeatPumpAndRefrigerationCycle.VapourCompMVR},
        default_topology="cascade",
        duties=frozenset({"heat_pump"}),
        options_model=MvrOptions,
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
    )
)
register_cycle(
    HprCycleSpec(
        id="brayton",
        label="Brayton cycle",
        topologies={"single": HeatPumpAndRefrigerationCycle.Brayton},
        default_topology="single",
        duties=DUTIES,
        options_model=BraytonOptions,
        available=False,
        all_periods=False,
    )
)


__all__ = [
    "DUTIES",
    "BraytonOptions",
    "CarnotOptions",
    "HprCycleSpec",
    "MvrOptions",
    "VapourCompressionOptions",
    "cycle_spec",
    "register_cycle",
    "registered_cycles",
    "spec_for_cycle",
]
