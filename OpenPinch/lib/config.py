"""Configuration defaults and global numerical constants for OpenPinch."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

from .config_metadata import (
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


class _HprConfig(SimpleNamespace):
    """HPR runtime settings plus derived backend-facing values."""

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


class Configuration:
    """Runtime configuration translated from flat user-facing option keys."""

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
    def options_catalog(cls):
        """Return declarative metadata for all supported flat option keys."""
        from ..classes._pinch_workspace.views import configuration_field_metadata

        return configuration_field_metadata()

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
        if period_id is not None:
            if period_id not in period_lookup:
                raise ValueError(f"Unknown period_id {period_id!r}.")
            resolved_idx = period_lookup[period_id]
            resolved_period = period_id
        elif period_idx is not None:
            resolved_idx = int(period_idx)
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

    def _build_groups(self, values: dict[str, Any]) -> None:
        groups: dict[str, SimpleNamespace] = {}
        for name, spec in CONFIG_FIELD_SPECS.items():
            group, field = spec.config_path
            group_obj = groups.setdefault(
                group, _HprConfig() if group == "hpr" else SimpleNamespace()
            )
            setattr(group_obj, field, deepcopy(values[name]))
        for group, group_obj in groups.items():
            setattr(self, group, group_obj)
