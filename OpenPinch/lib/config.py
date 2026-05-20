"""Configuration defaults and global numerical constants for OpenPinch.

The :class:`Configuration` object centralizes option flags and numerical
settings used across direct integration, utility targeting, and optional
advanced routines such as heat pump and cost targeting.
"""

from __future__ import annotations

from copy import deepcopy

from .config_metadata import CONFIG_FIELD_SPECS

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

    _LEGACY_OPTION_GATEWAYS = {"main", "turbine"}
    _RENAMED_OPTIONS = {
        "HP_CONDESATE": "IS_HIGH_P_COND_FLASH",
        "IS_HP_CONDESATE": "IS_HIGH_P_COND_FLASH",
        "IS_HP_CONDENSATE": "IS_HIGH_P_COND_FLASH",
        "CONDENSATE_FLASH_CORRECTION": "IS_HIGH_P_COND_FLASH",
    }

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
            if key == "REFRIGERANTS":
                ref_ls = (
                    value.replace(";", ",").split(",")
                    if isinstance(value, str)
                    else list(value)
                )
                setattr(self, key, ref_ls)
                continue
            setattr(self, key, value)

    @classmethod
    def _known_option_keys(cls) -> set[str]:
        """Return the supported configuration keys accepted by ``options``."""
        return {key for key in cls.__annotations__ if not key.startswith("_")}

    @classmethod
    def _validate_option_keys(cls, options: dict) -> dict:
        """Fail fast on unsupported workbook gateways and unknown option names."""
        legacy_gateways = sorted(
            key
            for key in options
            if key in cls._LEGACY_OPTION_GATEWAYS or str(key).startswith("PROP_TOP_")
        )
        if legacy_gateways:
            raise ValueError(
                "Legacy workbook option gateways are no longer supported: "
                f"{', '.join(legacy_gateways)}. Set canonical turbine fields directly "
                "on zone.config or pass them through Configuration(options=...), e.g. "
                "TURB_T_IN, TURB_P_IN, MIN_EFF, LOAD_FRACTION, ETA_MECH, TURB_MODEL, "
                "and IS_HIGH_P_COND_FLASH."
            )

        renamed_keys = sorted(key for key in options if key in cls._RENAMED_OPTIONS)
        if renamed_keys:
            rename_map = ", ".join(
                f"{key} -> {cls._RENAMED_OPTIONS[key]}" for key in renamed_keys
            )
            raise ValueError(
                "Unsupported configuration option name(s): "
                f"{rename_map}. Use the canonical zone.config field names instead."
            )

        known_keys = cls._known_option_keys()
        unknown_keys = sorted(key for key in options if key not in known_keys)
        if unknown_keys:
            raise ValueError(
                f"Unknown configuration option(s): {', '.join(unknown_keys)}."
            )

        return options


Configuration.__annotations__ = {
    name: spec.annotation for name, spec in CONFIG_FIELD_SPECS.items()
}
for _name, _spec in CONFIG_FIELD_SPECS.items():
    setattr(Configuration, _name, deepcopy(_spec.default))
