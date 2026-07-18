"""Focused edge tests for target, reporting, and I/O schemas."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import OpenPinch.contracts.input as io
import OpenPinch.contracts.reporting as reporting
import OpenPinch.domain.targets as targets
from OpenPinch.domain.enums import FluidPhase, StreamType, TargetType
from OpenPinch.domain.enums import ProblemTableLabel as ProblemTableLabel
from OpenPinch.domain.problem_table import ProblemTable
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection
from OpenPinch.domain.value import Value
from OpenPinch.presentation.reporting.results import serialize_target, target_to_result
from tests.support.paths import FIXTURES_ROOT

FIXTURE_PATH = FIXTURES_ROOT / "target_reporting_schema_cases.json"


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _utility_stream(spec: dict) -> Stream:
    return Stream(
        name=spec["name"],
        supply_temperature=spec["t_supply"],
        target_temperature=spec["t_target"],
        heat_flow=spec["heat_flow"],
        price=spec["price"],
        is_process_stream=False,
    )


def _utility_collection(*specs: dict) -> StreamCollection:
    collection = StreamCollection()
    for spec in specs:
        collection.add(_utility_stream(spec))
    return collection


def _problem_table() -> ProblemTable:
    return ProblemTable(
        {
            ProblemTableLabel.T: [100.0],
            ProblemTableLabel.H_HOT: [1.0],
            ProblemTableLabel.H_COLD: [0.0],
        }
    )


def test_io_schemas_normalise_fluid_fields_and_reject_edges():
    default_stream = io.StreamSchema(
        zone="Zone",
        name="Default",
        t_supply=90.0,
        t_target=40.0,
        heat_flow=5.0,
        fluid_name=None,
        fluid_phase=None,
    )
    assert default_stream.fluid_name is None
    assert default_stream.fluid_phase is None

    stream = io.StreamSchema(
        zone="Zone",
        name="S1",
        t_supply=100.0,
        t_target=50.0,
        heat_flow=10.0,
        fluid_name=" water ",
        fluid_phase=" ",
    )
    assert stream.fluid_name == "water"
    assert stream.fluid_phase is None

    utility = io.UtilitySchema(
        name="Steam",
        type=StreamType.Hot.value,
        t_supply=200.0,
        fluid_name=" ",
        fluid_phase=FluidPhase.liq,
    )
    assert utility.fluid_name is None
    assert utility.fluid_phase == FluidPhase.liq.value
    default_utility = io.UtilitySchema(
        name="Default utility",
        type=StreamType.Cold.value,
        t_supply=30.0,
        fluid_name=None,
        fluid_phase=None,
    )
    assert default_utility.fluid_name is None
    assert default_utility.fluid_phase is None

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        io.StreamSchema(
            zone="Zone",
            name="S1",
            stream_name="legacy",
            t_supply=100.0,
            t_target=50.0,
            heat_flow=10.0,
        )
    with pytest.raises(ValidationError, match="finite non-negative"):
        io.ZoneTreeSchema(name="Site", type="Site", dt_cont_multiplier=-1.0)
    with pytest.raises(ValueError, match="provided as a dict"):
        io.TargetInput._validate_options([])
    zone_tree = io.ZoneTreeSchema(name="Site", type="Site", dt_cont_multiplier=None)
    assert zone_tree.dt_cont_multiplier is None
    assert io.ZoneTreeSchema(
        name="Site",
        type="Site",
        dt_cont_multiplier=1,
    ).dt_cont_multiplier == pytest.approx(1.0)
    assert io.TargetInput(
        streams=[],
        options={"THERMAL_DT_CONT": 7.0},
    ).options == {"THERMAL_DT_CONT": 7.0}
    assert io.TargetInput(streams=[], options=None).options is None


def test_reporting_value_coercion_accepts_value_mapping_and_dumpable_objects():
    dumpable = SimpleNamespace(
        model_dump=lambda mode: {"value": 1.0, "unit": "kW"},
    )
    unit_object = SimpleNamespace(value=2.0, unit="kW")

    assert reporting.HeatUtility(name="Steam", heat_flow=Value(1.0, "kW")).heat_flow
    assert reporting.PinchTemp(cold_temp=dumpable).cold_temp.unit == "kW"
    assert reporting.PinchTemp(hot_temp={"value": 25.0, "unit": "degC"}).hot_temp.unit
    assert reporting.PinchTemp(hot_temp=unit_object).hot_temp.unit == "kW"
    assert reporting.PinchTemp(cold_temp=25.0).cold_temp.unit == "degC"


def test_target_name_validation_and_base_graph_helpers():
    with pytest.raises(ValueError, match="type is required"):
        targets._normalise_target_name(zone_name="Zone", target_type=None, name=None)
    with pytest.raises(ValueError, match="zone_name is required"):
        targets._normalise_target_name(
            zone_name="", target_type=TargetType.DI.value, name=None
        )
    with pytest.raises(ValueError, match="zone_name or name"):
        targets._normalise_target_name(
            zone_name=None, target_type=TargetType.DI.value, name=None
        )

    assert (
        targets._normalise_target_name(
            zone_name="Zone/Direct Integration",
            target_type=TargetType.DI.value,
            name=None,
        )
        == "Zone/Direct Integration"
    )
    assert (
        targets._normalise_target_name(
            zone_name="Zone",
            target_type=TargetType.DI.value,
            name="Custom",
        )
        == "Custom"
    )
    assert targets.BaseTargetModel._set_name("raw") == "raw"

    base = targets.BaseTargetModel(zone_name="Zone", type=TargetType.DI.value)
    with pytest.raises(NotImplementedError):
        target_to_result(base)

    graph_target = targets.GraphBackedTarget(zone_name="Zone", type=TargetType.ET.value)
    graph_target.add_graph("gcc", {"points": []})
    assert graph_target.graphs == {"gcc": {"points": []}}


def test_utility_summary_and_direct_target_reporting_round_trip():
    fixture = _fixture()
    hot_utilities = _utility_collection(fixture["hot_utility"])
    cold_utilities = _utility_collection(fixture["cold_utility"])
    target_data = fixture["direct_target"]
    target = targets.DirectIntegrationTarget(
        zone_name="Zone",
        type=TargetType.DI.value,
        pt=_problem_table(),
        pt_real=_problem_table(),
        hot_utilities=hot_utilities,
        cold_utilities=cold_utilities,
        **target_data,
    )

    assert len(target.utility_streams) == 2
    assert target.calc_utility_cost() == pytest.approx(110.0)

    result = target_to_result(target, isTotal=True)
    assert result.row_type == "footer"
    assert result.degree_of_integration.value == pytest.approx(50.0)
    assert result.hot_utilities[0].heat_flow.value == pytest.approx(1000.0)
    assert result.work_target.value == pytest.approx(20.0)
    assert result.area.value == pytest.approx(42.0)
    assert result.num_units == 3
    assert result.capital_cost.value == pytest.approx(10000.0)

    serialised = serialize_target(target, isTotal=True)
    assert serialised["Qh"] == {"value": 1000.0, "unit": "kW"}
    assert serialised["hot_utilities"][0]["heat_flow"]["unit"] == "kW"

    total_process = targets.TotalProcessTarget(
        zone_name="Zone",
        type=TargetType.TZ.value,
        hot_utility_target=1.0,
        cold_utility_target=2.0,
        heat_recovery_target=3.0,
    )
    assert target_to_result(total_process).Qh.value == pytest.approx(1.0)


def test_total_site_and_heat_pump_target_reporting_include_special_fields():
    fixture = _fixture()
    total_site = targets.TotalSiteTarget(
        zone_name="Zone",
        type=TargetType.TS.value,
        pt=_problem_table(),
        hot_utility_target=10.0,
        cold_utility_target=5.0,
        heat_recovery_target=20.0,
        work_target=3.0,
        turbine_efficiency_target=0.4,
    )
    total_site_result = target_to_result(total_site)
    assert total_site_result.work_target.value == pytest.approx(3.0)
    assert total_site_result.turbine_efficiency_target.value == pytest.approx(40.0)

    hpr = targets.DirectHeatPumpTarget(
        zone_name="Zone",
        type=TargetType.DHP.value,
        pt=_problem_table(),
        hpr_hot_streams=StreamCollection(),
        hpr_cold_streams=StreamCollection(),
        hpr_details={"fixture": True},
        **fixture["hpr_target"],
    )
    hpr_result = target_to_result(hpr)

    assert hpr_result.hpr_cycle == "fixture-cycle"
    assert hpr_result.hpr_utility_total.value == pytest.approx(120.0)
    assert hpr_result.hpr_work.value == pytest.approx(30.0)
    assert hpr_result.hpr_eta_he.value == pytest.approx(50.0)
    assert hpr_result.hpr_success is True
