from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import OpenPinch.analysis.heat_pumps.process_mvr as process_mvr_helpers
from OpenPinch.analysis.heat_pumps._process_mvr import (
    replacement_streams,
    selection,
    values,
)
from OpenPinch.analysis.heat_pumps._process_mvr.state import (
    _ProcessMVRStreamRecord,
    _StreamMembership,
)
from OpenPinch.analysis.heat_pumps.direct_mvr.models import DirectGasMVRStageResult
from OpenPinch.analysis.heat_pumps.process_mvr import ProcessMVRComponent
from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.value import Value
from OpenPinch.domain.zone import Zone
from tests.support.paths import FIXTURES_ROOT

PROCESS_MVR_EDGE_CASES = FIXTURES_ROOT / "process_mvr_edge_cases.json"


def _process_mvr_edge_cases():
    return json.loads(PROCESS_MVR_EDGE_CASES.read_text())


def _stage_result_from_fixture(name: str) -> DirectGasMVRStageResult:
    payload = _process_mvr_edge_cases()
    stage_payload = dict(payload[name])
    profile = np.asarray(payload["stage_profile"], dtype=float)
    return DirectGasMVRStageResult(
        **stage_payload,
        th_curve=profile.copy(),
        linearised_profile=profile.copy(),
    )


def _problem_payload(
    *, duplicate_display_names: bool = False, multiperiod: bool = False
):
    def period_valued(values, unit=None):
        payload = {"values": values}
        if unit is not None:
            payload["unit"] = unit
        return payload

    vapour_heat = period_valued([100.0, 130.0], "kW") if multiperiod else 100.0
    cold_heat = period_valued([120.0, 150.0], "kW") if multiperiod else 120.0
    streams = [
        {
            "zone": "Site/Evaporation Train",
            "name": "Evaporator vapour",
            "t_supply": period_valued([120.0, 125.0], "degC") if multiperiod else 120.0,
            "t_target": period_valued([80.0, 82.0], "degC") if multiperiod else 80.0,
            "p_supply": period_valued([101.325, 101.325], "kPa")
            if multiperiod
            else 101.325,
            "heat_flow": vapour_heat,
            "dt_cont": 0.0,
            "htc": 1.0,
            "fluid_name": "Air",
            "fluid_phase": "gas",
        },
        {
            "zone": "Site/Evaporation Train",
            "name": "Product heating",
            "t_supply": 30.0,
            "t_target": 100.0,
            "heat_flow": cold_heat,
            "dt_cont": 0.0,
            "htc": 1.0,
        },
    ]
    children = [{"name": "Evaporation Train", "type": "Process Zone"}]
    if duplicate_display_names:
        children.append({"name": "Second Train", "type": "Process Zone"})
        streams.extend(
            [
                {
                    "zone": "Site/Second Train",
                    "name": "Evaporator vapour",
                    "t_supply": 118.0,
                    "t_target": 78.0,
                    "p_supply": 101.325,
                    "heat_flow": 90.0,
                    "dt_cont": 0.0,
                    "htc": 1.0,
                    "fluid_name": "Air",
                    "fluid_phase": "gas",
                },
                {
                    "zone": "Site/Second Train",
                    "name": "Second product heating",
                    "t_supply": 35.0,
                    "t_target": 96.0,
                    "heat_flow": 95.0,
                    "dt_cont": 0.0,
                    "htc": 1.0,
                },
            ]
        )
    return {
        "streams": streams,
        "utilities": [],
        "zone_tree": {"name": "Site", "type": "Site", "children": children},
        "options": {"PROBLEM_PERIOD_IDS": ["0", "peak"]} if multiperiod else {},
    }


def _problem(**kwargs) -> PinchProblem:
    return PinchProblem(_problem_payload(**kwargs), project_name="Site")


def test_add_process_mvr_applies_replacements_and_registers_component():
    pytest.importorskip("CoolProp")
    problem = _problem()
    zone = problem.master_zone.get_subzone("Evaporation Train")
    source = next(
        stream for stream in zone.hot_streams if stream.name == "Evaporator vapour"
    )

    component = problem.add_component.process_mvr(
        source,
        n_stages=2,
        liquid_injection=False,
        mvr_stage_t_lift=10.0,
    )

    assert isinstance(component, ProcessMVRComponent)
    assert component.id == "mvr_1"
    assert problem.process_components["mvr_1"] is component
    assert source.active is False
    assert len(component.replacement_streams) >= 2
    assert all(stream.active for stream in component.replacement_streams)
    assert all(
        float(stream.p_supply) > float(source.p_supply)
        for stream in component.replacement_streams
    )
    assert min(float(stream.t_target) for stream in component.replacement_streams) == (
        pytest.approx(float(source.t_target), abs=0.05)
    )
    assert "Site/Evaporation Train" in component.affected_zone_paths
    assert any(stream is source for _key, stream in zone.hot_streams.items())
    assert any(
        ".mvr_1.Evaporator vapour_direct_MVR_H1" in key
        for key, _stream in zone.hot_streams.items()
    )


def test_process_mvr_can_select_by_display_name_and_qualified_key():
    pytest.importorskip("CoolProp")
    by_name = _problem()
    component = by_name.add_component.process_mvr(
        "Evaporator vapour",
        liquid_injection=False,
    )
    assert len(component.original_streams) == 1


def test_process_mvr_can_use_pressure_ratio_target():
    pytest.importorskip("CoolProp")
    problem = _problem()

    component = problem.add_component.process_mvr(
        "Evaporator vapour",
        liquid_injection=False,
        mvr_stage_pressure_ratio=1.2,
    )

    stage = component.stage_results_by_period["0"][0]
    assert stage.p_out / stage.p_in == pytest.approx(1.2)


def test_process_mvr_rejects_conflicting_compression_targets():
    problem = _problem()

    with pytest.raises(
        ValueError,
        match="either mvr_stage_t_lift or mvr_stage_pressure_ratio",
    ):
        problem.add_component.process_mvr(
            "Evaporator vapour",
            mvr_stage_t_lift=10.0,
            mvr_stage_pressure_ratio=1.2,
        )

    by_key = _problem()
    zone = by_key.master_zone.get_subzone("Evaporation Train")
    key = next(
        key
        for key, stream in zone.hot_streams.items()
        if stream.name == "Evaporator vapour"
    )
    component = by_key.add_component.process_mvr(key, liquid_injection=False)
    assert len(component.original_streams) == 1


@pytest.mark.parametrize("n_stages", [0, -1, 1.5, True])
def test_process_mvr_rejects_invalid_stage_count(n_stages):
    problem = _problem()

    with pytest.raises(ValueError, match="positive integer"):
        problem.add_component.process_mvr(
            "Evaporator vapour",
            n_stages=n_stages,
            liquid_injection=False,
        )


def test_process_mvr_rejects_empty_source_selector_list():
    problem = _problem()

    with pytest.raises(ValueError, match="at least one stream selector"):
        problem.add_component.process_mvr([], liquid_injection=False)


def test_process_mvr_rejects_conflicting_state_contexts():
    problem = _problem(multiperiod=True)

    with pytest.raises(ValueError, match="period_id and options\\['period_id'\\]"):
        problem.add_component.process_mvr(
            "Evaporator vapour",
            period_id="peak",
            options={"period_id": "0"},
            liquid_injection=False,
        )


def test_duplicate_display_names_apply_to_all_matching_hot_gas_streams():
    pytest.importorskip("CoolProp")
    problem = _problem(duplicate_display_names=True)

    component = problem.add_component.process_mvr(
        "Evaporator vapour",
        liquid_injection=False,
    )

    assert len(component.original_streams) == 2
    assert all(not stream.active for stream in component.original_streams)
    assert len(component.replacement_streams) >= 2


def test_process_mvr_activate_deactivate_toggles_recorded_streams_and_invalidates():
    pytest.importorskip("CoolProp")
    problem = _problem()
    zone = problem.master_zone.get_subzone("Evaporation Train")
    component = problem.add_component.process_mvr(
        "Evaporator vapour",
        liquid_injection=False,
    )
    first_target = problem.target.direct_heat_integration(zone_name="Evaporation Train")
    assert first_target.heat_recovery_target == pytest.approx(120.0)
    assert first_target.process_component_work_target == pytest.approx(
        component.work_for_zone(zone, period_id="0", period_idx=0)
    )
    assert first_target.work_target == pytest.approx(
        first_target.process_component_work_target
    )
    assert problem.results is not None

    component.deactivate()

    assert component.active is False
    assert all(stream.active for stream in component.original_streams)
    assert all(not stream.active for stream in component.replacement_streams)
    assert problem.results is None
    second_target = problem.target.direct_heat_integration(
        zone_name="Evaporation Train"
    )
    assert second_target.heat_recovery_target == pytest.approx(100.0)
    assert second_target.process_component_work_target == pytest.approx(0.0)
    assert second_target.work_target is None

    component.activate()

    assert component.active is True
    assert all(not stream.active for stream in component.original_streams)
    assert all(stream.active for stream in component.replacement_streams)


def test_process_mvr_is_not_a_targeting_method():
    problem = _problem()

    assert not hasattr(problem.target, "process_mvr")
    assert not hasattr(problem, "add")


def test_process_mvr_rejects_duplicate_component_id():
    pytest.importorskip("CoolProp")
    problem = _problem()
    problem.add_component.process_mvr(
        "Evaporator vapour", mvr_id="vapour_mvr", liquid_injection=False
    )

    with pytest.raises(ValueError, match="already exists"):
        problem.add_component.process_mvr("Evaporator vapour", mvr_id="vapour_mvr")


def test_process_mvr_rejects_non_gas_stream():
    payload = _problem_payload()
    payload["streams"][0]["fluid_phase"] = "vle"
    problem = PinchProblem(payload, project_name="Site")

    with pytest.raises(ValueError, match="fluid_phase='gas' or fluid_phase='vapour'"):
        problem.add_component.process_mvr("Evaporator vapour")


def test_process_mvr_accepts_vapour_stream_and_derives_saturation_pressure():
    pytest.importorskip("CoolProp")
    payload = _problem_payload()
    payload["streams"][0].update(
        {
            "t_supply": 100.0,
            "t_target": 99.5,
            "p_supply": None,
            "fluid_name": "Water",
            "fluid_phase": "vapour",
            "heat_flow": 100.0,
        }
    )
    problem = PinchProblem(payload, project_name="Site")

    component = problem.add_component.process_mvr(
        "Evaporator vapour",
        mvr_stage_pressure_ratio=1.2,
        liquid_injection=False,
    )

    source = component.original_streams[0]
    assert float(source.p_supply) == pytest.approx(101.4, rel=0.02)
    assert float(source.p_target) == pytest.approx(float(source.p_supply))
    assert component.stage_results_by_period["0"][0].work > 0.0


def test_process_mvr_rejects_vapour_stream_away_from_saturation_pressure():
    pytest.importorskip("CoolProp")
    payload = _problem_payload()
    payload["streams"][0].update(
        {
            "t_supply": 100.0,
            "t_target": 99.5,
            "p_supply": 80.0,
            "fluid_name": "Water",
            "fluid_phase": "vapour",
        }
    )
    problem = PinchProblem(payload, project_name="Site")

    with pytest.raises(ValueError, match="saturation pressure"):
        problem.add_component.process_mvr("Evaporator vapour")


def test_process_mvr_defaults_missing_gas_target_pressure():
    pytest.importorskip("CoolProp")
    payload = _problem_payload()
    payload["streams"][0]["p_target"] = None
    problem = PinchProblem(payload, project_name="Site")

    component = problem.add_component.process_mvr(
        "Evaporator vapour",
        liquid_injection=False,
    )

    assert float(component.original_streams[0].p_target) == pytest.approx(
        float(component.original_streams[0].p_supply)
    )


def test_process_mvr_rejects_subcritical_liquid_like_gas_state():
    pytest.importorskip("CoolProp")
    payload = _problem_payload()
    payload["streams"][0].update(
        {
            "t_supply": 120.0,
            "t_target": 80.0,
            "p_supply": 300.0,
            "p_target": 300.0,
            "fluid_name": "Water",
            "fluid_phase": "gas",
        }
    )
    problem = PinchProblem(payload, project_name="Site")

    with pytest.raises(ValueError, match="above saturation pressure"):
        problem.add_component.process_mvr("Evaporator vapour")


def test_process_mvr_rejects_cold_stream_selection():
    problem = _problem()

    with pytest.raises(ValueError, match="must be a hot stream"):
        problem.add_component.process_mvr("Product heating")


def test_process_mvr_solves_all_periods_for_multiperiod_streams():
    pytest.importorskip("CoolProp")
    problem = _problem(multiperiod=True)

    component = problem.add_component.process_mvr(
        "Evaporator vapour",
        liquid_injection=False,
        period_id="peak",
    )

    assert set(component.stage_results_by_period) == {"0", "peak"}
    replacement = component.replacement_streams[0]
    assert len(component.replacement_streams) == 1
    assert replacement.has_segments
    assert all(segment.t_supply.num_periods == 2 for segment in replacement.segments)
    assert [segment.segment_index for segment in replacement.segments] == list(
        range(replacement.segment_count)
    )
    assert float(replacement.heat_flow[0]) > 0.0
    assert float(replacement.heat_flow[1]) > 0.0
    assert min(
        float(stream.t_target[0]) for stream in component.replacement_streams
    ) == (pytest.approx(80.0, abs=0.05))
    assert min(
        float(stream.t_target[1]) for stream in component.replacement_streams
    ) == (pytest.approx(82.0, abs=0.05))

    normal_target = problem.target.direct_heat_integration(
        zone_name="Evaporation Train",
        period_id="0",
    )
    peak_target = problem.target.direct_heat_integration(
        zone_name="Evaporation Train",
        period_id="peak",
    )
    peak_target_by_idx = problem.target.direct_heat_integration(
        zone_name="Evaporation Train",
        options={"period_idx": 1},
    )

    assert normal_target.heat_recovery_target != peak_target.heat_recovery_target
    assert normal_target.process_component_work_target == pytest.approx(
        component.work_for_zone(
            problem.master_zone.get_subzone("Evaporation Train"),
            period_id="0",
            period_idx=0,
        )
    )
    assert peak_target.process_component_work_target == pytest.approx(
        component.work_for_zone(
            problem.master_zone.get_subzone("Evaporation Train"),
            period_id="peak",
            period_idx=1,
        )
    )
    assert normal_target.process_component_work_target != pytest.approx(
        peak_target.process_component_work_target
    )
    assert peak_target_by_idx.heat_recovery_target == pytest.approx(
        peak_target.heat_recovery_target
    )
    assert peak_target_by_idx.process_component_work_target == pytest.approx(
        peak_target.process_component_work_target
    )


def test_process_mvr_work_for_zone_uses_active_state_membership_and_period_context():
    root = Zone("Site")
    child = Zone("Area", parent_zone=root)
    root.add_zone(child)
    other = Zone("Other", parent_zone=root)
    root.add_zone(other)
    source = Stream(
        name="HotGas",
        t_supply=120.0,
        t_target=80.0,
        p_supply=101.325,
        heat_flow=100.0,
        fluid_name="Air",
        fluid_phase="gas",
    )
    record = _ProcessMVRStreamRecord(
        original_stream=source,
        original_memberships=[_StreamMembership(zone=child, key="Area.HotGas")],
        replacement_streams=[],
        replacement_memberships=[],
        stage_results_by_period={
            "base": [SimpleNamespace(work=3.0), SimpleNamespace(work=4.0)],
            "peak": [SimpleNamespace(work=11.0)],
        },
        period_label_by_index={0: "base", 1: "peak"},
    )
    problem = SimpleNamespace(master_zone=root, _results="cached")
    component = ProcessMVRComponent(
        id="mvr_1",
        problem=problem,
        stream_records=[record],
    )

    assert component.work_for_zone(other, period_id="base") == 0.0
    assert component.work_for_zone(root, period_id="base") == pytest.approx(7.0)
    assert component.work_for_zone(child, period_idx=1) == pytest.approx(11.0)
    assert component.work_for_zone(child, period_id="missing") == pytest.approx(7.0)

    component.active = False

    assert component.work_for_zone(child, period_id="base") == 0.0


def test_process_mvr_helper_guards_and_period_value_edges():
    source = Stream(
        name="HotGas",
        t_supply=120.0,
        t_target=80.0,
        p_supply=101.325,
        heat_flow=100.0,
        fluid_name="Air",
        fluid_phase="gas",
    )

    with pytest.raises(ValueError, match="source_streams is required"):
        selection.normalise_source_selectors(None)
    with pytest.raises(ValueError, match="is not in a zone"):
        replacement_streams.build_process_mvr_stream_record(
            source_stream=source,
            memberships=[],
            settings=SimpleNamespace(n_stages=1),
            period_ids=None,
            num_periods=None,
        )

    assert selection.normalise_source_selectors(source) == [source]
    assert selection.normalise_source_selectors(("H1", "H2")) == [
        "H1",
        "H2",
    ]
    assert values.period_values_or_scalar([4.0]) == 4.0
    assert values.period_values_or_scalar([4.0, 5.0]) == [4.0, 5.0]
    assert values.required_period_value(
        Value([293.15], "K"),
        0,
        "t_supply",
        "HotGas",
    ) == pytest.approx(20.0)
    with pytest.raises(ValueError, match="requires heat_flow"):
        values.required_period_value(None, 0, "heat_flow", "HotGas")
    assert values.value_at_index(12.5, 3) == pytest.approx(12.5)
    assert values.value_at_index(Value(12.0, "kW"), 3) == pytest.approx(12.0)
    assert values.value_at_index(None, 0) is None


def test_process_mvr_stream_matching_and_source_validation_errors():
    root = Zone("Site")
    hot = Stream(
        name="HotGas",
        t_supply=120.0,
        t_target=80.0,
        p_supply=101.325,
        heat_flow=100.0,
        fluid_name="Air",
        fluid_phase="gas",
    )
    cold = Stream(
        name="ColdLoad",
        t_supply=20.0,
        t_target=80.0,
        heat_flow=100.0,
    )
    root.hot_streams.add(hot, key="H1")
    root.cold_streams.add(cold, key="C1")

    with pytest.raises(ValueError, match="No active hot gas"):
        selection.match_source_streams(root, [])
    with pytest.raises(ValueError, match="must be a hot stream"):
        selection.match_one_selector(root, "ColdLoad")
    with pytest.raises(ValueError, match="was not found"):
        selection.match_one_selector(root, "Missing")

    inactive = Stream(
        name="InactiveGas",
        t_supply=120.0,
        t_target=80.0,
        p_supply=101.325,
        heat_flow=100.0,
        fluid_name="Air",
        fluid_phase="gas",
    )
    inactive.active = False
    utility = Stream(
        name="UtilityGas",
        t_supply=120.0,
        t_target=80.0,
        p_supply=101.325,
        heat_flow=100.0,
        is_process_stream=False,
        fluid_name="Air",
        fluid_phase="gas",
    )
    no_fluid = Stream(
        name="NoFluid",
        t_supply=120.0,
        t_target=80.0,
        p_supply=101.325,
        heat_flow=100.0,
        fluid_phase="gas",
    )

    with pytest.raises(ValueError, match="is not active"):
        selection.validate_process_mvr_source(inactive, "InactiveGas")
    with pytest.raises(ValueError, match="must be a process stream"):
        selection.validate_process_mvr_source(utility, "UtilityGas")
    with pytest.raises(ValueError, match="must be a hot stream"):
        selection.validate_process_mvr_source(cold, "ColdLoad")
    with pytest.raises(ValueError, match="requires fluid_name"):
        selection.validate_process_mvr_source(no_fluid, "NoFluid")


def test_process_mvr_profile_stage_and_component_id_helpers():
    positive_stage = _stage_result_from_fixture("positive_stage")
    zero_delta_stage = _stage_result_from_fixture("zero_delta_stage")
    assert replacement_streams.stage_mass_flow(zero_delta_stage) == 0.0
    assert replacement_streams.stage_mass_flow(positive_stage) == pytest.approx(0.5)
    assert replacement_streams.stage_segment_heat_flow(
        positive_stage,
        300.0,
        200.0,
    ).value == pytest.approx(50.0)

    problem = SimpleNamespace(process_components={"mvr_1": object()})

    assert process_mvr_helpers._resolve_component_id(problem, None) == "mvr_2"
    assert process_mvr_helpers._resolve_component_id(problem, "custom") == "custom"
    with pytest.raises(ValueError, match="already exists"):
        process_mvr_helpers._resolve_component_id(problem, "mvr_1")


def test_process_mvr_phase_validation_edges_use_real_fluid_data():
    pytest.importorskip("CoolProp")
    heating_stream = Stream(
        name="HeatingGas",
        t_supply=80.0,
        t_target=120.0,
        p_supply=101.325,
        heat_flow=100.0,
        fluid_name="Air",
        fluid_phase="gas",
    )
    gas_without_pressure = Stream(
        name="NoPressureGas",
        t_supply=120.0,
        t_target=80.0,
        p_supply=None,
        heat_flow=100.0,
        fluid_name="Air",
        fluid_phase="gas",
    )
    vapour_without_pressure = Stream(
        name="WaterVapour",
        t_supply=[100.0, 99.0],
        t_target=[99.5, 98.5],
        p_supply=None,
        p_target=None,
        heat_flow=[100.0, 90.0],
        fluid_name="Water",
        fluid_phase="vapour",
    )

    with pytest.raises(ValueError, match="must cool from supply to target"):
        selection.validate_process_mvr_source_phase(
            heating_stream,
            "HeatingGas",
        )
    with pytest.raises(ValueError, match="requires p_supply"):
        selection.validate_process_mvr_source_phase(
            gas_without_pressure,
            "NoPressureGas",
        )

    selection.validate_process_mvr_source_phase(
        vapour_without_pressure,
        "WaterVapour",
    )

    assert list(vapour_without_pressure.p_supply.value) == pytest.approx(
        list(vapour_without_pressure.p_target.value)
    )
    assert len(vapour_without_pressure.p_supply.value) == 2

    with pytest.raises(ValueError, match="critical temperature"):
        selection.validate_vapour_state(
            selector="WaterVapour",
            fluid="Water",
            t_supply=400.0,
            p_supply=None,
            t_crit=373.0,
        )
    saturated_pressure = selection.validate_vapour_state(
        selector="WaterVapour",
        fluid="Water",
        t_supply=100.0,
        p_supply=None,
        t_crit=373.0,
    )
    assert saturated_pressure == pytest.approx(101.4, rel=0.02)
    assert selection.validate_vapour_state(
        selector="WaterVapour",
        fluid="Water",
        t_supply=100.0,
        p_supply=saturated_pressure,
        t_crit=373.0,
    ) == pytest.approx(saturated_pressure)
    with pytest.raises(ValueError, match="saturation pressure"):
        selection.validate_vapour_state(
            selector="WaterVapour",
            fluid="Water",
            t_supply=100.0,
            p_supply=80.0,
            t_crit=373.0,
        )

    with pytest.raises(ValueError, match="above critical pressure"):
        selection.validate_gas_or_supercritical_state(
            selector="WaterGas",
            fluid="Water",
            t_c=100.0,
            p_kpa=30000.0,
            t_crit=373.0,
            p_crit=22064.0,
            state_label="supply",
        )
    selection.validate_gas_or_supercritical_state(
        selector="WaterGas",
        fluid="Water",
        t_c=400.0,
        p_kpa=30000.0,
        t_crit=373.0,
        p_crit=22064.0,
        state_label="supply",
    )
    selection.validate_gas_or_supercritical_state(
        selector="WaterGas",
        fluid="Water",
        t_c=400.0,
        p_kpa=100.0,
        t_crit=373.0,
        p_crit=22064.0,
        state_label="target",
    )
    with pytest.raises(ValueError, match="above saturation pressure"):
        selection.validate_gas_or_supercritical_state(
            selector="WaterGas",
            fluid="Water",
            t_c=100.0,
            p_kpa=150.0,
            t_crit=373.0,
            p_crit=22064.0,
            state_label="target",
        )
    with pytest.raises(ValueError, match="not available in CoolProp"):
        selection.coolprop_value("TCRIT", "NotAFluid")
