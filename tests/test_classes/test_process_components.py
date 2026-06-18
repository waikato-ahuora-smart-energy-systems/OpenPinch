from __future__ import annotations

import pytest

from OpenPinch.classes.pinch_problem import PinchProblem
from OpenPinch.services.components.process_mvr import ProcessMVRComponent


def _problem_payload(
    *, duplicate_display_names: bool = False, multistate: bool = False
):
    def stateful(values, unit=None):
        payload = {"values": values}
        if unit is not None:
            payload["unit"] = unit
        return payload

    vapour_heat = stateful([100.0, 130.0], "kW") if multistate else 100.0
    cold_heat = stateful([120.0, 150.0], "kW") if multistate else 120.0
    streams = [
        {
            "zone": "Site/Evaporation Train",
            "name": "Evaporator vapour",
            "t_supply": stateful([120.0, 125.0], "degC") if multistate else 120.0,
            "t_target": stateful([80.0, 82.0], "degC") if multistate else 80.0,
            "p_supply": stateful([101.325, 101.325], "kPa") if multistate else 101.325,
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
        "options": {"STATE_IDS": ["0", "peak"]} if multistate else {},
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
        ".mvr_1.Evaporator vapour_direct_MVR_H1_S" in key
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

    stage = component.stage_results_by_state["0"][0]
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
    problem = _problem(multistate=True)

    with pytest.raises(ValueError, match="state_id and options\\['state_id'\\]"):
        problem.add_component.process_mvr(
            "Evaporator vapour",
            state_id="peak",
            options={"state_id": "0"},
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
        component.work_for_zone(zone, state_id="0", state_idx=0)
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
    assert component.stage_results_by_state["0"][0].work > 0.0


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


def test_process_mvr_solves_all_states_for_multistate_streams():
    pytest.importorskip("CoolProp")
    problem = _problem(multistate=True)

    component = problem.add_component.process_mvr(
        "Evaporator vapour",
        liquid_injection=False,
        state_id="peak",
    )

    assert set(component.stage_results_by_state) == {"0", "peak"}
    replacement = component.replacement_streams[0]
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
        state_id="0",
    )
    peak_target = problem.target.direct_heat_integration(
        zone_name="Evaporation Train",
        state_id="peak",
    )
    peak_target_by_idx = problem.target.direct_heat_integration(
        zone_name="Evaporation Train",
        options={"idx": 1},
    )

    assert normal_target.heat_recovery_target != peak_target.heat_recovery_target
    assert normal_target.process_component_work_target == pytest.approx(
        component.work_for_zone(
            problem.master_zone.get_subzone("Evaporation Train"),
            state_id="0",
            state_idx=0,
        )
    )
    assert peak_target.process_component_work_target == pytest.approx(
        component.work_for_zone(
            problem.master_zone.get_subzone("Evaporation Train"),
            state_id="peak",
            state_idx=1,
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
