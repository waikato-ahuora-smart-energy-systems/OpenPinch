"""Configuration defaults and global numerical constants for OpenPinch."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from types import SimpleNamespace, UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from ._value.coercion import coerce_period_index
from .configuration_fields import (
    CONFIG_FIELD_SPECS,
    input_unit_options_to_map,
    output_unit_options_to_map,
    validate_configuration_options,
)

C_to_K: float = 273.15  # degrees
tol: float = 1e-6
T_CRIT: float = 373.9  # C
ACTIVATE_TIMING = False
LOG_TIMING = False

__all__ = [
    "ACTIVATE_TIMING",
    "C_to_K",
    "Configuration",
    "LOG_TIMING",
    "T_CRIT",
    "tol",
]


# Typed views of the configuration groups. ``CONFIG_FIELD_SPECS`` stays the
# single source of defaults and validation; these classes only declare each
# group's attribute names and types statically so readers and type checkers
# see them. ``_check_group_types`` fails at import if the two drift apart.
# All groups are frozen: change settings with :meth:`Configuration.update_values`,
# which updates the flat values and rebuilds the groups, so a later rebuild
# cannot undo the change.
_frozen_group = dataclass(frozen=True, slots=True)


@_frozen_group
class ProblemConfig:
    """Problem shape: top zone and period definitions."""

    top_zone_name: str
    top_zone_identifier: str
    period_ids: list[str]
    period_weights: list[float]


@_frozen_group
class InputUnitsConfig:
    """Units assumed for user-supplied input values."""

    temperature: str
    pressure: str
    enthalpy: str
    heat_flow: str
    delta_t: str
    htc: str
    price: str


@_frozen_group
class OutputUnitsConfig:
    """Units used when reporting results."""

    heat_flow: str
    temperature: str
    percent: str
    utility_cost: str
    work: str
    area: str
    capital_cost: str
    annual_cost: str
    exergy: str
    hpr_cop: str


@_frozen_group
class ReportingConfig:
    """Result formatting settings."""

    decimal_places: int


@_frozen_group
class EnvironmentConfig:
    """Ambient (dead-state) conditions."""

    temperature: float
    pressure: float


@_frozen_group
class ThermalConfig:
    """Heat-transfer approach temperatures and coefficients."""

    dt_cont: float
    dt_phase_change: float
    htc: float


@_frozen_group
class DirectConfig:
    """Direct integration settings."""

    balanced_cc_enabled: bool
    vertical_gcc_enabled: bool
    assisted_ht_enabled: bool
    assisted_ht_dt: float


@_frozen_group
class CostingConfig:
    """Utility, heat exchanger and HPR costing parameters."""

    utility_price: float
    annual_op_time: float
    hx_unit_cost: float
    hx_area_coeff: float
    hx_area_exp: float
    discount_rate: float
    service_life: float
    hpr_ele_price: float
    hpr_price_ratio_heat_to_ele: float
    hpr_price_ratio_cooling_water_to_ele: float
    hpr_cooling_water_temperature: float
    hpr_cooling_water_dt_min: float
    hpr_refrigeration_eta_ii: float
    hpr_refrigeration_dt: float
    hpr_equipment_cost: float
    hpr_installation_factor: float
    hpr_cost_exp: float
    hpr_cost_fixed_share: float
    hpr_cost_stage_share: float
    hpr_cost_temp_factor: float
    hpr_cost_temp_base: float
    hpr_capital_recovery_enabled: bool
    hpr_utility_capital_recovery_enabled: bool
    hpr_hot_utility_capital_cost: float
    hpr_refrigeration_capital_cost: float


@_frozen_group
class HensConfig:
    """Heat exchanger network synthesis settings."""

    approach_temperatures: list[float]
    dt_cont_multipliers: list[float] | None
    derivative_thresholds: list[float]
    synthesis_quality_tier: int
    pdm_stage_pair_limit: int | None
    tdm_parent_limit: int | None
    stage_packing: str
    stage_selection: list[int]
    solver_pdm: str
    solver_tdm: str
    solver_evm: str
    solver_options_pdm: dict[str, Any]
    solver_options_tdm: dict[str, Any]
    solver_options_evm: dict[str, Any]
    solve_tolerance: float
    max_parallel: int
    evm_n_ad_branches: int | None
    evm_n_rm_branches: int | None
    evm_beam_width: int
    log_level: str
    output_folder: str
    output_formats: list[str]
    run_id: str
    best_solutions_to_save: int


@_frozen_group
class HprConfig:
    """HPR runtime settings plus derived backend-facing values."""

    type: str
    load_mode: str
    load_fraction: float
    load_duty: float | None
    load_period_values: dict[str, float]
    multiperiod_optimization_enabled: bool
    refrigerants: list[str]
    refrigerant_sort_enabled: bool
    mvr_fluids: list[str]
    mvr_count: int
    mvr_eta_comp: float
    mvr_eta_motor: float
    n_cond: int
    n_evap: int
    eta_comp: float
    eta_exp: float
    eta_ii_carnot: float
    he_eta_ii_carnot: float
    integrated_expander_enabled: bool
    dt_cont: float
    dt_ihx: float
    dt_cascade_hx: float
    dt_env_cont: float
    max_multistart: int
    eta_penalty: float
    rho_penalty: float
    bb_minimiser: str
    initialise_simulated_cycle: bool

    @staticmethod
    def _normalise_config_list(values, *, uppercase: bool = False) -> list[str]:
        normalised = [str(value).strip() for value in values if str(value).strip()]
        if uppercase:
            return [value.upper() for value in normalised]
        return normalised

    @property
    def normalised_refrigerants(self) -> list[str]:
        """Return HPR refrigerant names normalised for thermodynamic backends."""
        return self._normalise_config_list(self.refrigerants, uppercase=True)

    @property
    def normalised_mvr_fluids(self) -> list[str]:
        """Return MVR fluid names normalised for thermodynamic backends."""
        return self._normalise_config_list(self.mvr_fluids) or ["Water"]

    @property
    def effective_eta_ii_he_carnot(self) -> float:
        """Return the usable Carnot heat-engine efficiency for HPR targeting."""
        return float(self.he_eta_ii_carnot) if self.integrated_expander_enabled else 0.0


@_frozen_group
class ProcessMvrConfig:
    """Direct process MVR efficiencies."""

    eta_comp: float
    eta_motor: float


@_frozen_group
class PowerConfig:
    """Power cogeneration (steam turbine) settings."""

    turbine_work_enabled: bool
    turb_t_in: float
    turb_p_in: float
    min_eff: float
    load_fraction: float
    eta_mech: float
    turb_model: str
    high_p_cond_flash_enabled: bool


_GROUP_TYPES: dict[str, type] = {
    "problem": ProblemConfig,
    "input_units": InputUnitsConfig,
    "output_units": OutputUnitsConfig,
    "reporting": ReportingConfig,
    "environment": EnvironmentConfig,
    "thermal": ThermalConfig,
    "direct": DirectConfig,
    "costing": CostingConfig,
    "hens": HensConfig,
    "hpr": HprConfig,
    "process_mvr": ProcessMvrConfig,
    "power": PowerConfig,
}


def _canonical_annotation(annotation: Any) -> Any:
    """Return a comparable form that treats ``List``/``list`` and unions alike."""
    origin = get_origin(annotation)
    if origin is None:
        return annotation
    if origin is UnionType:
        origin = Union
    return origin, tuple(_canonical_annotation(arg) for arg in get_args(annotation))


def _check_group_types() -> None:
    """Fail if the group classes disagree with ``CONFIG_FIELD_SPECS``."""
    expected: dict[str, list[tuple[str, Any]]] = {}
    for spec in CONFIG_FIELD_SPECS.values():
        group, field = spec.config_path
        expected.setdefault(group, []).append(
            (field, _canonical_annotation(spec.annotation))
        )
    problems = sorted(set(expected) ^ set(_GROUP_TYPES))
    for group in set(expected) & set(_GROUP_TYPES):
        hints = get_type_hints(_GROUP_TYPES[group])
        declared = [
            (field.name, _canonical_annotation(hints[field.name]))
            for field in fields(_GROUP_TYPES[group])
        ]
        if declared != expected[group]:
            problems.append(group)
    if problems:
        raise TypeError(
            "Configuration group classes do not match CONFIG_FIELD_SPECS for: "
            + ", ".join(problems)
        )


_check_group_types()


class Configuration:
    """Runtime configuration translated from flat user-facing option keys."""

    problem: ProblemConfig
    input_units: InputUnitsConfig
    output_units: OutputUnitsConfig
    reporting: ReportingConfig
    environment: EnvironmentConfig
    thermal: ThermalConfig
    direct: DirectConfig
    costing: CostingConfig
    hens: HensConfig
    hpr: HprConfig
    process_mvr: ProcessMvrConfig
    power: PowerConfig

    def __init__(
        self,
        options: dict | None = None,
        top_zone_name: str = CONFIG_FIELD_SPECS["PROBLEM_TOP_ZONE_NAME"].default,
        top_zone_identifier: str = CONFIG_FIELD_SPECS[
            "PROBLEM_TOP_ZONE_IDENTIFIER"
        ].default,
    ):
        """Initialise defaults and optionally apply validated flat options."""
        if options is not None and not isinstance(options, dict):
            raise TypeError("Configuration options must be provided as a dict.")

        values = {
            name: deepcopy(spec.default) for name, spec in CONFIG_FIELD_SPECS.items()
        }
        values["PROBLEM_TOP_ZONE_NAME"] = top_zone_name
        values["PROBLEM_TOP_ZONE_IDENTIFIER"] = top_zone_identifier

        if options:
            values.update(self._validate_options(options))

        self._values = values
        self._build_groups(values)

    @classmethod
    def from_options(
        cls,
        options: dict | None = None,
        *,
        top_zone_name: str = CONFIG_FIELD_SPECS["PROBLEM_TOP_ZONE_NAME"].default,
        top_zone_identifier: str = CONFIG_FIELD_SPECS[
            "PROBLEM_TOP_ZONE_IDENTIFIER"
        ].default,
    ) -> "Configuration":
        """Build a runtime configuration from flat user-facing options."""
        return cls(
            options=options,
            top_zone_name=top_zone_name,
            top_zone_identifier=top_zone_identifier,
        )

    @classmethod
    def _known_option_keys(cls) -> set[str]:
        """Return the supported flat configuration keys accepted by ``options``."""
        return set(CONFIG_FIELD_SPECS)

    @classmethod
    def _validate_options(cls, options: dict) -> dict:
        """Fail fast on unsupported keys and invalid option values."""
        return validate_configuration_options(options)

    def for_period(
        self,
        period_id: str | None = None,
        period_idx: int | None = None,
    ):
        """Return a lightweight period context for this configuration."""
        period_ids = list(self.problem.period_ids)
        period_lookup = {period: index for index, period in enumerate(period_ids)}
        try:
            explicit_idx = (
                None if period_idx is None else coerce_period_index(period_idx)
            )
        except IndexError as exc:
            raise ValueError(f"Unknown period index {period_idx!r}.") from exc
        if period_id is not None:
            if period_id not in period_lookup:
                raise ValueError(f"Unknown period_id {period_id!r}.")
            resolved_idx = period_lookup[period_id]
            resolved_period = period_id
            if explicit_idx is not None and explicit_idx != resolved_idx:
                raise ValueError(
                    f"period_id {period_id!r} resolves to period_idx {resolved_idx}, "
                    f"but period_idx {explicit_idx} was also provided."
                )
        elif explicit_idx is not None:
            resolved_idx = explicit_idx
            try:
                resolved_period = period_ids[resolved_idx]
            except IndexError as exc:
                raise ValueError(f"Unknown period index {period_idx!r}.") from exc
        else:
            resolved_idx = 0
            resolved_period = period_ids[0] if period_ids else None
        weight = (
            self.problem.period_weights[resolved_idx]
            if resolved_idx < len(self.problem.period_weights)
            else 1.0
        )
        return SimpleNamespace(
            period_id=resolved_period,
            period_idx=resolved_idx,
            weight=weight,
        )

    @property
    def input_unit_overrides(self) -> dict[str, str]:
        """Return input unit overrides in the unit-system mapping format."""
        return input_unit_options_to_map(self._values)

    @property
    def output_unit_overrides(self) -> dict[str, str]:
        """Return output unit overrides in the unit-system mapping format."""
        return output_unit_options_to_map(self._values)

    def update_values(self, **values: Any) -> None:
        """Set flat option values (e.g. ``COSTING_ANNUAL_OP_TIME=8760``) in place.

        The flat values are the source the typed groups are built from, so the
        change survives any later group rebuild. Values are deep-copied, so later
        changes to the caller's objects do not leak in, and are not revalidated.
        """
        unknown = sorted(set(values) - set(CONFIG_FIELD_SPECS))
        if unknown:
            raise KeyError(f"Unknown configuration option(s): {', '.join(unknown)}")
        self._values.update(deepcopy(values))
        self._build_groups(self._values)

    def _build_groups(self, values: dict[str, Any]) -> None:
        """(Re)build the typed group objects from flat option ``values``."""
        group_values: dict[str, dict[str, Any]] = {group: {} for group in _GROUP_TYPES}
        for name, spec in CONFIG_FIELD_SPECS.items():
            group, field = spec.config_path
            group_values[group][field] = deepcopy(values[name])
        for group, group_type in _GROUP_TYPES.items():
            setattr(self, group, group_type(**group_values[group]))
