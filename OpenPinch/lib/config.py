"""Configuration defaults and global numerical constants for OpenPinch.

The :class:`Configuration` object centralizes option flags and numerical
settings used across direct integration, utility targeting, and optional
advanced routines such as heat pump and cost targeting.
"""

from __future__ import annotations

from copy import deepcopy

from .config_metadata import (
    CONFIG_FIELD_SPECS,
    configuration_option_status,
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


class Configuration:
    """Runtime configuration defaults used throughout OpenPinch.

    The attributes on this class combine global numerical settings, workbook-
    compatible feature flags, and advanced-analysis parameters such as heat pump
    or costing options. A ``Configuration`` instance is attached to each
    :class:`~OpenPinch.classes.zone.Zone` so workflows can vary behaviour by
    hierarchy level if needed.
    """

    def __init__(
        self,
        options: dict | None = None,
        top_zone_name: str = CONFIG_FIELD_SPECS["TOP_ZONE_NAME"].default,
        top_zone_identifier: str = CONFIG_FIELD_SPECS["TOP_ZONE_IDENTIFIER"].default,
    ):
        """Initialise defaults and optionally apply user-provided options."""
        for key in type(self).__annotations__:
            if key.startswith("_"):
                continue
            setattr(self, key, deepcopy(getattr(type(self), key)))

        self.TOP_ZONE_NAME = top_zone_name
        self.TOP_ZONE_IDENTIFIER = top_zone_identifier

        if options is None:
            return

        if not isinstance(options, dict):
            raise TypeError("Configuration options must be provided as a dict.")

        for key, value in self._validate_option_keys(options).items():
            spec = CONFIG_FIELD_SPECS[str(key)]
            if key in {"REFRIGERANTS", "MVR_FLUIDS"}:
                ref_ls = (
                    value.replace(";", ",").split(",")
                    if isinstance(value, str)
                    else list(value)
                )
                setattr(self, key, ref_ls)
                continue
            if spec.enum_cls is not None:
                value = self._validate_enum_option(str(key), value, spec.enum_cls)
            setattr(self, key, value)

    @classmethod
    def _known_option_keys(cls) -> set[str]:
        """Return the supported configuration keys accepted by ``options``."""
        return {key for key in cls.__annotations__ if not key.startswith("_")}

    @classmethod
    def _validate_option_keys(cls, options: dict) -> dict:
        """Fail fast on unsupported or removed configuration keys."""
        statuses = {key: configuration_option_status(str(key)) for key in options}

        dead_keys = sorted(
            key for key, status in statuses.items() if status.runtime_status == "dead"
        )
        if dead_keys:
            raise ValueError(
                f"Unknown configuration option(s): {', '.join(dead_keys)}."
            )

        return options

    @staticmethod
    def _validate_enum_option(key: str, value, enum_cls):
        """Return the enum value for a supported config enum option."""
        if isinstance(value, enum_cls):
            return value.value
        allowed = {item.value for item in enum_cls}
        if value not in allowed:
            allowed_str = ", ".join(sorted(str(item) for item in allowed))
            raise ValueError(
                f"Invalid value for configuration option {key}: {value!r}. "
                f"Allowed values are: {allowed_str}."
            )
        return value


Configuration.__annotations__ = {
    name: spec.annotation for name, spec in CONFIG_FIELD_SPECS.items()
}
for _name, _spec in CONFIG_FIELD_SPECS.items():
    setattr(Configuration, _name, deepcopy(_spec.default))
