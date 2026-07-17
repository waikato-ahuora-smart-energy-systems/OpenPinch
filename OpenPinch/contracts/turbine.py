"""Schemas for turbine solve outputs."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TurbineStageResult(BaseModel):
    """Detailed result for one solved turbine stage."""

    stage: int
    source_index: int
    stage_type: str
    temperature: float
    process_duty: float
    pressure_in: float
    pressure_out: float
    mass_flow_in: float
    mass_flow_extracted: float
    mass_flow_out: float
    enthalpy_in: float
    enthalpy_out: float
    condensate_enthalpy: float
    saturation_enthalpy: float
    dh_isentropic: float
    work_actual: float
    work_isentropic: float
    isentropic_efficiency: float
    turbine_model: str


class TurbineSolveResult(BaseModel):
    """Validated output for a multi-stage turbine targeting solve."""

    model_config = ConfigDict(extra="forbid")

    mode: str
    turbine_model: str
    load_frac: float
    mech_eff: float
    min_eff: float
    is_high_p_cond_flash: bool
    total_work: float
    total_isentropic_work: float
    overall_efficiency: float
    total_process_duty: float
    steam_mass_flow_in: Optional[float] = None
    inlet_pressure: Optional[float] = None
    inlet_temperature: Optional[float] = None
    sink_pressure: Optional[float] = None
    sink_temperature: Optional[float] = None
    stage_temperatures: List[float] = Field(default_factory=list)
    stage_heat_flows: List[float] = Field(default_factory=list)
    stages: List[TurbineStageResult] = Field(default_factory=list)


__all__ = ["TurbineSolveResult", "TurbineStageResult"]
