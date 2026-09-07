"""Immutable numerical handoff from HPR targeting to residual utility studies."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _FrozenRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class HPRLoadSummary(_FrozenRecord):
    """Selected service and achieved cycle duty, in kW."""

    mode: Literal["heat_pump", "refrigeration"]
    available: float = Field(ge=0)
    selected: float = Field(ge=0)
    achieved: float = Field(ge=0)
    cycle_heating: float = Field(ge=0)
    cycle_cooling: float = Field(ge=0)
    work: float
    ambient_hot: float = Field(ge=0)
    ambient_cold: float = Field(ge=0)
    heat_flow_unit: Literal["kW"] = "kW"


class HPRResidualProfile(_FrozenRecord):
    """Normalized net cascade and positive utility demands on one exact grid."""

    temperatures: tuple[float, ...]
    net: tuple[float, ...]
    heating: tuple[float, ...]
    cooling: tuple[float, ...]
    temperature_basis: Literal["shifted", "real"]
    temperature_unit: Literal["C"] = "C"
    heat_flow_unit: Literal["kW"] = "kW"

    @model_validator(mode="after")
    def _validate_grid(self):
        n = len(self.temperatures)
        if n < 2 or any(len(v) != n for v in (self.net, self.heating, self.cooling)):
            raise ValueError("Residual profiles require one aligned nonempty grid.")
        if any(a <= b for a, b in zip(self.temperatures, self.temperatures[1:])):
            raise ValueError("Residual temperatures must strictly decrease.")
        if any(
            v < -1e-7
            for values in (self.net, self.heating, self.cooling)
            for v in values
        ):
            raise ValueError("Residual utility demands must be nonnegative.")
        return self


class HPRThermalSlice(_FrozenRecord):
    """Physical thermal boundary segment, including fixed HPR and ambient ports."""

    side: Literal["hot", "cold"]
    supply_temperature: float = Field(gt=-273.15)
    target_temperature: float = Field(gt=-273.15)
    duty: float = Field(ge=0)


class HPRResidualData(_FrozenRecord):
    """Finite residual and physical-temperature boundary for one selected period."""

    profile: HPRResidualProfile
    period_id: str
    mode: Literal["heat_pump", "refrigeration"]
    physical_temperatures: tuple[float, ...]
    physical_hot_composite: tuple[float, ...]
    physical_cold_composite: tuple[float, ...]
    thermal_slices: tuple[HPRThermalSlice, ...]

    @model_validator(mode="after")
    def _validate_physical_grid(self):
        n = len(self.physical_temperatures)
        if (
            n < 2
            or len(self.physical_hot_composite) != n
            or len(self.physical_cold_composite) != n
        ):
            raise ValueError("Physical boundary composites must share a nonempty grid.")
        if any(
            a <= b
            for a, b in zip(self.physical_temperatures, self.physical_temperatures[1:])
        ):
            raise ValueError("Physical boundary temperatures must strictly decrease.")
        if not self.period_id.strip():
            raise ValueError("Residual period identity must be nonempty.")
        return self


class HPRResidualSnapshot(_FrozenRecord):
    """Detached numerical basis and its source provenance."""

    data: HPRResidualData
    source_identity: str
    source_fingerprint: str
    source_zone: str
    result_digest: str
