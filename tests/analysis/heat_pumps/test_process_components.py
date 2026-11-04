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
from OpenPinch.application.workspace import PinchWorkspace
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

    component = problem.components.add_process_mvr(
        source,
        n_stages=2,
        liquid_injection=False,
        mvr_stage_t_lift=10.0,
    )

    assert isinstance(component, ProcessMVRComponent)
    assert component.id == "mvr_1"
    assert problem.process_components["mvr_1"] is component
    assert source.is_active is True  # The original observation stays detached.
    zone = problem.master_zone.get_subzone("Evaporation Train")
    assert (
        next(
            stream for stream in zone.hot_streams if stream.name == source.name
        ).is_active
        is False
    )
    assert len(component.replacement_streams) >= 2
    assert all(stream.is_active for stream in component.replacement_streams)
    assert all(
        float(stream.supply_pressure) > float(source.supply_pressure)
        for stream in component.replacement_streams
    )
    assert min(
        float(stream.target_temperature) for stream in component.replacement_streams
    ) == (pytest.approx(float(source.target_temperature), abs=0.05))
    assert "Site/Evaporation Train" in component.affected_zone_paths
    assert any(stream.name == source.name for _key, stream in zone.hot_streams.items())
    assert any(
        ".mvr_1.Evaporator vapour_direct_MVR_H1" in key
        for key, _stream in zone.hot_streams.items()
    )


def test_process_mvr_can_select_by_display_name_and_qualified_key():
    pytest.importorskip("CoolProp")
    by_name = _problem()
    component = by_name.components.add_process_mvr(
        "Evaporator vapour",
        liquid_injection=False,
    )
    assert len(component.original_streams) == 1


def test_process_mvr_can_use_pressure_ratio_target():
    pytest.importorskip("CoolProp")
    problem = _problem()

    component = problem.components.add_process_mvr(
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
        problem.components.add_process_mvr(
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
    component = by_key.components.add_process_mvr(key, liquid_injection=False)
    assert len(component.original_streams) == 1


@pytest.mark.parametrize("n_stages", [0, -1, 1.5, True])
def test_process_mvr_rejects_invalid_stage_count(n_stages):
    problem = _problem()

    with pytest.raises(ValueError, match="positive integer"):
        problem.components.add_process_mvr(
            "Evaporator vapour",
            n_stages=n_stages,
            liquid_injection=False,
        )


def test_process_mvr_rejects_empty_source_selector_list():
    problem = _problem()

    with pytest.raises(ValueError, match="at least one stream selector"):
        problem.components.add_process_mvr([], liquid_injection=False)


def test_process_mvr_rejects_conflicting_state_contexts():
    problem = _problem(multiperiod=True)

    with pytest.raises(ValueError, match="period_id and options\\['period_id'\\]"):
        problem.components.add_process_mvr(
            "Evaporator vapour",
            period_id="peak",
            options={"period_id": "0"},
            liquid_injection=False,
        )


def test_duplicate_display_names_apply_to_all_matching_hot_gas_streams():
    pytest.importorskip("CoolProp")
    problem = _problem(duplicate_display_names=True)

    component = problem.components.add_process_mvr(
        "Evaporator vapour",
        liquid_injection=False,
    )

    assert len(component.original_streams) == 2
    assert all(not stream.is_active for stream in component.original_streams)
    assert len(component.replacement_streams) >= 2


def test_process_mvr_activate_deactivate_toggles_recorded_streams_and_invalidates():
    pytest.importorskip("CoolProp")
    problem = _problem()
    zone = problem.master_zone.get_subzone("Evaporation Train")
    component = problem.components.add_process_mvr(
        "Evaporator vapour",
        liquid_injection=False,
    )
    first_target = problem.target.direct_heat_integration(zone="Evaporation Train")
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
    assert all(stream.is_active for stream in component.original_streams)
    assert all(not stream.is_active for stream in component.replacement_streams)
    assert problem.results is None
    second_target = problem.target.direct_heat_integration(zone="Evaporation Train")
    assert second_target.heat_recovery_target == pytest.approx(100.0)
    assert second_target.process_component_work_target == pytest.approx(0.0)
    assert second_target.work_target is None

    component.activate()

    assert component.active is True


def test_add_process_mvr_accepts_engineering_efficiency_names():
    pytest.importorskip("CoolProp")
    problem = _problem()

    component = problem.components.add_process_mvr(
        "Evaporator vapour",
        liquid_injection=False,
        compressor_efficiency=0.73,
        motor_efficiency=0.96,
    )

    assert component.settings.eta_mvr_comp == pytest.approx(0.73)
    assert component.settings.eta_motor == pytest.approx(0.96)
    assert all(not stream.is_active for stream in component.original_streams)
    assert all(stream.is_active for stream in component.replacement_streams)


def test_process_mvr_is_not_a_targeting_method():
    problem = _problem()

    assert not hasattr(problem.target, "process_mvr")
    assert not hasattr(problem, "add")


def test_process_mvr_rejects_duplicate_component_id():
    pytest.importorskip("CoolProp")
    problem = _problem()
    problem.components.add_process_mvr(
        "Evaporator vapour", mvr_id="vapour_mvr", liquid_injection=False
    )

    with pytest.raises(ValueError, match="already exists"):
        problem.components.add_process_mvr("Evaporator vapour", mvr_id="vapour_mvr")


def test_process_mvr_rejects_non_gas_stream():
    payload = _problem_payload()
    payload["streams"][0]["fluid_phase"] = "vle"
    problem = PinchProblem(payload, project_name="Site")

    with pytest.raises(ValueError, match="fluid_phase='gas' or fluid_phase='vapour'"):
        problem.components.add_process_mvr("Evaporator vapour")


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

    component = problem.components.add_process_mvr(
        "Evaporator vapour",
        mvr_stage_pressure_ratio=1.2,
        liquid_injection=False,
    )

    source = component.original_streams[0]
    assert float(source.supply_pressure) == pytest.approx(101.4, rel=0.02)
    assert float(source.target_pressure) == pytest.approx(float(source.supply_pressure))
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
        problem.components.add_process_mvr("Evaporator vapour")


def test_process_mvr_defaults_missing_gas_target_pressure():
    pytest.importorskip("CoolProp")
    payload = _problem_payload()
    payload["streams"][0]["p_target"] = None
    problem = PinchProblem(payload, project_name="Site")

    component = problem.components.add_process_mvr(
        "Evaporator vapour",
        liquid_injection=False,
    )

    assert float(component.original_streams[0].target_pressure) == pytest.approx(
        float(component.original_streams[0].supply_pressure)
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
        problem.components.add_process_mvr("Evaporator vapour")


def test_process_mvr_rejects_cold_stream_selection():
    problem = _problem()

    with pytest.raises(ValueError, match="must be a hot stream"):
        problem.components.add_process_mvr("Product heating")


def _vapour_payload(
    *, fluid, t_supply, t_target, heat_flow, p_supply=None, fluid_phase="gas"
):
    """Return a process MVR case; list arguments give one value per period."""
    multiperiod = isinstance(heat_flow, list)
    if p_supply is None:
        p_supply = [101.325] * len(heat_flow) if multiperiod else 101.325

    def valued(values, unit):
        return {"values": values, "unit": unit} if multiperiod else values

    return {
        "streams": [
            {
                "zone": "Site/Train",
                "name": "Vapour",
                "t_supply": valued(t_supply, "degC"),
                "t_target": valued(t_target, "degC"),
                "p_supply": valued(p_supply, "kPa"),
                "heat_flow": valued(heat_flow, "kW"),
                "dt_cont": 0.0,
                "htc": 1.0,
                "fluid_name": fluid,
                "fluid_phase": fluid_phase,
            },
            {
                "zone": "Site/Train",
                "name": "Product",
                "t_supply": 30.0,
                "t_target": 100.0,
                "heat_flow": 120.0,
                "dt_cont": 0.0,
                "htc": 1.0,
            },
        ],
        "utilities": [],
        "zone_tree": {
            "name": "Site",
            "type": "Site",
            "children": [{"name": "Train", "type": "Process Zone"}],
        },
        "options": {"PROBLEM_PERIOD_IDS": ["0", "peak"]} if multiperiod else {},
    }


@pytest.mark.parametrize(
    ("fluid", "t_supply", "t_target", "heat_flow", "injection_applied"),
    [
        # Air stays dry: its periods share breakpoints that differ only by rounding.
        ("Air", [120.0, 125.0], [80.0, 82.0], [100.0, 130.0], [False, False]),
    ],
    ids=["dry_air"],
)
def test_two_period_process_mvr_default_matches_each_period_alone(
    fluid, t_supply, t_target, heat_flow, injection_applied
):
    pytest.importorskip("CoolProp")
    from CoolProp.CoolProp import PropsSI

    problem = PinchProblem(
        _vapour_payload(
            fluid=fluid, t_supply=t_supply, t_target=t_target, heat_flow=heat_flow
        ),
        project_name="Site",
    )

    component = problem.components.add_process_mvr(
        "Vapour", mvr_stage_pressure_ratio=1.5, motor_efficiency=0.95
    )

    replacement = component.replacement_streams[0]
    for index, period_id in enumerate(("0", "peak")):
        stage = component.stage_results_by_period[period_id][0]
        assert stage.liquid_injection_applied is injection_applied[index]

        alone = PinchProblem(
            _vapour_payload(
                fluid=fluid,
                t_supply=t_supply[index],
                t_target=t_target[index],
                heat_flow=heat_flow[index],
            ),
            project_name="Site",
        ).components.add_process_mvr(
            "Vapour", mvr_stage_pressure_ratio=1.5, motor_efficiency=0.95
        )
        alone_stream = alone.replacement_streams[0]
        alone_stage = next(iter(alone.stage_results_by_period.values()))[0]
        assert stage.work == pytest.approx(alone_stage.work, rel=1e-12)
        assert float(replacement.heat_flow[index]) == pytest.approx(
            float(alone_stream.heat_flow[0]), rel=1e-9
        )
        assert float(replacement.supply_temperature[index]) == pytest.approx(
            float(alone_stream.supply_temperature[0]), abs=1e-6
        )
        assert float(replacement.target_temperature[index]) == pytest.approx(
            float(alone_stream.target_temperature[0]), abs=1e-6
        )

        # Energy balance: heat released is the source vapour's enthalpy after
        # compression (shaft work added), cooled to the target at discharge pressure.
        h_in = PropsSI(
            "H",
            "T",
            t_supply[index] + 273.15,
            "P",
            101325.0,
            fluid,
        )
        h_target = PropsSI(
            "H",
            "T",
            t_target[index] + 273.15,
            "P",
            1.5 * 101325.0,
            fluid,
        )
        shaft_kw = stage.work * 0.95
        released_kw = shaft_kw + stage.source_mass_flow * (h_in - h_target) / 1e3
        assert float(replacement.heat_flow[index]) == pytest.approx(
            released_kw, rel=1e-6
        )


def test_two_period_injected_water_matches_each_period_alone():
    pytest.importorskip("CoolProp")
    from CoolProp.CoolProp import PropsSI

    # Liquid injection needs a source that condenses, so saturated vapour.
    t_supply, t_target, heat_flow = [100.0, 105.0], [99.5, 104.5], [100.0, 130.0]
    p_supply = [
        PropsSI("P", "T", t + 273.15, "Q", 1.0, "Water") / 1e3 for t in t_supply
    ]
    payload = {"fluid": "Water", "fluid_phase": "vapour"}
    problem = PinchProblem(
        _vapour_payload(
            t_supply=t_supply,
            t_target=t_target,
            heat_flow=heat_flow,
            p_supply=p_supply,
            **payload,
        ),
        project_name="Site",
    )

    component = problem.components.add_process_mvr(
        "Vapour", mvr_stage_pressure_ratio=1.5, motor_efficiency=0.95
    )

    replacement = component.replacement_streams[0]
    for index, period_id in enumerate(("0", "peak")):
        stage = component.stage_results_by_period[period_id][0]
        assert stage.liquid_injection_applied is True

        alone = PinchProblem(
            _vapour_payload(
                t_supply=t_supply[index],
                t_target=t_target[index],
                heat_flow=heat_flow[index],
                p_supply=p_supply[index],
                **payload,
            ),
            project_name="Site",
        ).components.add_process_mvr(
            "Vapour", mvr_stage_pressure_ratio=1.5, motor_efficiency=0.95
        )
        alone_stream = alone.replacement_streams[0]
        alone_stage = next(iter(alone.stage_results_by_period.values()))[0]
        assert stage.work == pytest.approx(alone_stage.work, rel=1e-12)
        assert float(replacement.heat_flow[index]) == pytest.approx(
            float(alone_stream.heat_flow[0]), rel=1e-9
        )
        assert float(replacement.supply_temperature[index]) == pytest.approx(
            float(alone_stream.supply_temperature[0]), abs=1e-6
        )
        assert float(replacement.target_temperature[index]) == pytest.approx(
            float(alone_stream.target_temperature[0]), abs=1e-6
        )
        # The source duty already holds the latent heat. The tolerance covers the
        # small rise in liquid enthalpy from supply to discharge pressure.
        assert float(replacement.heat_flow[index]) == pytest.approx(
            heat_flow[index] + stage.work * 0.95, rel=1e-3
        )


def test_gas_source_condensing_at_stage_pressure_is_rejected_in_any_period():
    pytest.importorskip("CoolProp")
    problem = PinchProblem(
        _vapour_payload(
            fluid="Water",
            t_supply=[120.0, 125.0],
            t_target=[100.0, 102.0],
            heat_flow=[100.0, 130.0],
        ),
        project_name="Site",
    )

    with pytest.raises(ValueError, match="period 0, stage 1"):
        problem.components.add_process_mvr("Vapour", mvr_stage_pressure_ratio=1.5)


def test_process_mvr_solves_all_periods_for_multiperiod_streams():
    pytest.importorskip("CoolProp")
    problem = _problem(multiperiod=True)

    component = problem.components.add_process_mvr(
        "Evaporator vapour",
        liquid_injection=False,
        period_id="peak",
    )

    assert set(component.stage_results_by_period) == {"0", "peak"}
    replacement = component.replacement_streams[0]
    assert len(component.replacement_streams) == 1
    assert replacement.has_segments
    assert all(
        segment.supply_temperature.num_periods == 2 for segment in replacement.segments
    )
    assert [segment.segment_index for segment in replacement.segments] == list(
        range(replacement.segment_count)
    )
    assert float(replacement.heat_flow[0]) > 0.0
    assert float(replacement.heat_flow[1]) > 0.0
    assert min(
        float(stream.target_temperature[0]) for stream in component.replacement_streams
    ) == (pytest.approx(80.0, abs=0.05))
    assert min(
        float(stream.target_temperature[1]) for stream in component.replacement_streams
    ) == (pytest.approx(82.0, abs=0.05))

    normal_target = problem.target.direct_heat_integration(
        zone="Evaporation Train",
        period_id="0",
    )
    peak_target = problem.target.direct_heat_integration(
        zone="Evaporation Train",
        period_id="peak",
    )
    peak_target_by_idx = problem.target.direct_heat_integration(
        zone="Evaporation Train",
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
        supply_temperature=120.0,
        target_temperature=80.0,
        supply_pressure=101.325,
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
        supply_temperature=120.0,
        target_temperature=80.0,
        supply_pressure=101.325,
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
        supply_temperature=120.0,
        target_temperature=80.0,
        supply_pressure=101.325,
        heat_flow=100.0,
        fluid_name="Air",
        fluid_phase="gas",
    )
    cold = Stream(
        name="ColdLoad",
        supply_temperature=20.0,
        target_temperature=80.0,
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
        supply_temperature=120.0,
        target_temperature=80.0,
        supply_pressure=101.325,
        heat_flow=100.0,
        fluid_name="Air",
        fluid_phase="gas",
    )
    inactive.is_active = False
    utility = Stream(
        name="UtilityGas",
        supply_temperature=120.0,
        target_temperature=80.0,
        supply_pressure=101.325,
        heat_flow=100.0,
        is_process_stream=False,
        fluid_name="Air",
        fluid_phase="gas",
    )
    no_fluid = Stream(
        name="NoFluid",
        supply_temperature=120.0,
        target_temperature=80.0,
        supply_pressure=101.325,
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
        supply_temperature=80.0,
        target_temperature=120.0,
        supply_pressure=101.325,
        heat_flow=100.0,
        fluid_name="Air",
        fluid_phase="gas",
    )
    gas_without_pressure = Stream(
        name="NoPressureGas",
        supply_temperature=120.0,
        target_temperature=80.0,
        supply_pressure=None,
        heat_flow=100.0,
        fluid_name="Air",
        fluid_phase="gas",
    )
    vapour_without_pressure = Stream(
        name="WaterVapour",
        supply_temperature=[100.0, 99.0],
        target_temperature=[99.5, 98.5],
        supply_pressure=None,
        target_pressure=None,
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

    assert list(vapour_without_pressure.supply_pressure.value) == pytest.approx(
        list(vapour_without_pressure.target_pressure.value)
    )
    assert len(vapour_without_pressure.supply_pressure.value) == 2

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


def _hot_utility(problem: PinchProblem) -> float:
    return problem.target.direct_heat_integration().hot_utility_target


def _active_stream_flags(problem: PinchProblem) -> dict[str, bool]:
    return {
        key: stream.is_active
        for key, stream in problem._master_zone.hot_streams.items()
    }


@pytest.mark.parametrize(
    "change",
    [
        lambda problem: problem.update_options({"ENV_TEMPERATURE": 30.0}),
        lambda problem: problem.set_dt_cont_multiplier(2.0),
    ],
    ids=["update_options", "set_dt_cont_multiplier"],
)
def test_input_rebuild_replays_component_on_the_new_tree(change):
    pytest.importorskip("CoolProp")
    problem = _problem()
    component = problem.components.add_process_mvr(
        "Evaporator vapour", mvr_stage_pressure_ratio=1.5
    )

    change(problem)

    assert problem.process_components["mvr_1"] is component
    assert component.problem is problem
    root = problem._master_zone
    record = component.stream_records[0]
    assert any(stream is record.original_stream for stream in root.hot_streams)
    assert record.original_memberships[0].zone is root

    recreated = _problem()
    change(recreated)
    recreated_component = recreated.components.add_process_mvr(
        "Evaporator vapour", mvr_stage_pressure_ratio=1.5
    )
    assert component.work_for_zone(root) == pytest.approx(
        recreated_component.work_for_zone(recreated._master_zone), rel=1e-12
    )
    assert _hot_utility(problem) == pytest.approx(_hot_utility(recreated), rel=1e-9)


def test_kept_handle_controls_the_rebuilt_tree():
    pytest.importorskip("CoolProp")
    problem = _problem()
    without_mvr = _hot_utility(_problem())
    component = problem.components.add_process_mvr(
        "Evaporator vapour", mvr_stage_pressure_ratio=1.5
    )
    with_mvr = _hot_utility(problem)
    assert with_mvr != pytest.approx(without_mvr, rel=1e-6)

    problem.update_options({"ENV_TEMPERATURE": 30.0})
    component.deactivate()
    assert _hot_utility(problem) == pytest.approx(without_mvr, rel=1e-9)
    component.activate()
    assert _hot_utility(problem) == pytest.approx(with_mvr, rel=1e-9)


def test_replay_follows_changed_options_for_defaults_and_keeps_explicit_arguments():
    pytest.importorskip("CoolProp")
    problem = _problem()
    defaulted = problem.components.add_process_mvr(
        "Evaporator vapour", mvr_stage_pressure_ratio=1.5
    )
    other = _problem()
    explicit = other.components.add_process_mvr(
        "Evaporator vapour",
        mvr_stage_pressure_ratio=1.5,
        compressor_efficiency=0.6,
        motor_efficiency=0.9,
    )
    new_options = {"PROCESS_MVR_ETA_COMP": 0.9, "PROCESS_MVR_ETA_MOTOR": 0.8}
    assert defaulted.settings.eta_mvr_comp == pytest.approx(0.7, rel=1e-12)

    problem.update_options(new_options)
    other.update_options(new_options)

    assert defaulted.settings.eta_mvr_comp == pytest.approx(0.9, rel=1e-12)
    assert defaulted.settings.eta_motor == pytest.approx(0.8, rel=1e-12)
    assert defaulted.settings.mvr_stage_pressure_ratio == pytest.approx(1.5, rel=1e-12)
    assert explicit.settings.eta_mvr_comp == pytest.approx(0.6, rel=1e-12)
    assert explicit.settings.eta_motor == pytest.approx(0.9, rel=1e-12)

    recreated = _problem()
    recreated.update_options(new_options)
    recreated_component = recreated.components.add_process_mvr(
        "Evaporator vapour", mvr_stage_pressure_ratio=1.5
    )
    assert defaulted.work_for_zone(problem._master_zone) == pytest.approx(
        recreated_component.work_for_zone(recreated._master_zone), rel=1e-12
    )


def test_replay_keeps_a_deactivated_component_inactive():
    pytest.importorskip("CoolProp")
    problem = _problem()
    component = problem.components.add_process_mvr(
        "Evaporator vapour", mvr_stage_pressure_ratio=1.5
    )
    component.deactivate()

    problem.update_options({"ENV_TEMPERATURE": 30.0})

    assert component.active is False
    assert component.work_for_zone(problem._master_zone) == pytest.approx(
        0.0, abs=1e-12
    )
    record = component.stream_records[0]
    assert record.original_stream.is_active is True
    assert all(not stream.is_active for stream in record.replacement_streams)


def test_replay_restores_overlapping_components_exactly():
    pytest.importorskip("CoolProp")
    problem = _problem()
    first = problem.components.add_process_mvr(
        "Evaporator vapour", mvr_stage_pressure_ratio=1.5
    )
    first.deactivate()
    second = problem.components.add_process_mvr(
        "Evaporator vapour", mvr_stage_pressure_ratio=1.3
    )
    before = _active_stream_flags(problem)

    problem.update_options({"ENV_TEMPERATURE": 30.0})

    assert list(problem.process_components) == ["mvr_1", "mvr_2"]
    assert (first.active, second.active) == (False, True)
    assert _active_stream_flags(problem) == before


def test_replay_restores_a_component_chained_on_a_replacement_stream():
    pytest.importorskip("CoolProp")
    problem = _problem()
    first = problem.components.add_process_mvr(
        "Evaporator vapour", mvr_stage_pressure_ratio=1.5
    )
    second = problem.components.add_process_mvr(
        first.replacement_streams[0].name, mvr_stage_pressure_ratio=1.2
    )
    before = _active_stream_flags(problem)
    hot_utility = _hot_utility(problem)

    problem.update_options({"ENV_TEMPERATURE": 30.0})

    assert _active_stream_flags(problem) == before
    assert any(
        stream is second.stream_records[0].original_stream
        for stream in first.stream_records[0].replacement_streams
    )
    assert _hot_utility(problem) == pytest.approx(hot_utility, rel=1e-9)


def test_replay_finds_sources_after_a_project_rename():
    pytest.importorskip("CoolProp")
    problem = _problem(duplicate_display_names=True)
    component = problem.components.add_process_mvr(
        "Site/Second Train.Evaporator vapour", mvr_stage_pressure_ratio=1.5
    )
    problem.project_name = "Renamed"

    problem.update_options({"ENV_TEMPERATURE": 30.0})

    record = component.stream_records[0]
    assert any(
        stream is record.original_stream
        for stream in problem._master_zone.get_subzone("Second Train").hot_streams
    )


def test_replay_covers_every_period():
    pytest.importorskip("CoolProp")
    problem = _problem(multiperiod=True)
    component = problem.components.add_process_mvr(
        "Evaporator vapour", mvr_stage_pressure_ratio=1.5
    )
    work = {
        period_id: component.work_for_zone(problem._master_zone, period_id=period_id)
        for period_id in ("0", "peak")
    }

    problem.update_options({"ENV_TEMPERATURE": 30.0})

    assert set(component.stage_results_by_period) == {"0", "peak"}
    for period_id, expected in work.items():
        assert component.work_for_zone(
            problem._master_zone, period_id=period_id
        ) == pytest.approx(expected, rel=1e-12)


def test_failed_replay_raises_and_leaves_problem_and_handles_untouched(monkeypatch):
    pytest.importorskip("CoolProp")
    problem = _problem(duplicate_display_names=True)
    first = problem.components.add_process_mvr(
        "Site/Evaporation Train.Evaporator vapour", mvr_stage_pressure_ratio=1.5
    )
    second = problem.components.add_process_mvr(
        "Site/Second Train.Evaporator vapour", mvr_stage_pressure_ratio=1.5
    )
    root = problem._master_zone
    records = (first.stream_records, second.stream_records)
    settings = (first.settings, second.settings)
    real_build = process_mvr_helpers.build_process_mvr_stream_record

    def fail_on_second_source(**kwargs):
        if any(m.zone.name == "Second Train" for m in kwargs["memberships"]):
            raise ValueError("cannot solve under the new options")
        return real_build(**kwargs)

    monkeypatch.setattr(
        process_mvr_helpers, "build_process_mvr_stream_record", fail_on_second_source
    )

    with pytest.raises(ValueError, match="cannot solve under the new options"):
        problem.update_options({"PROCESS_MVR_ETA_COMP": 0.9})

    assert problem._master_zone is root
    assert "PROCESS_MVR_ETA_COMP" not in problem.to_problem_json()["options"]
    assert problem.process_components == {"mvr_1": first, "mvr_2": second}
    assert (first.stream_records, second.stream_records) == records
    assert (first.settings, second.settings) == settings
    assert first.settings.eta_mvr_comp == pytest.approx(0.7, rel=1e-12)


def test_replay_raises_when_a_source_stream_is_missing():
    pytest.importorskip("CoolProp")
    problem = _problem()
    component = problem.components.add_process_mvr(
        "Evaporator vapour", mvr_stage_pressure_ratio=1.5
    )
    component.stream_records[0].original_memberships[0].key = "missing"

    with pytest.raises(ValueError, match="mvr_1.*Evaporator vapour.*not found"):
        problem.update_options({"ENV_TEMPERATURE": 30.0})


def test_loading_a_new_source_still_discards_components():
    pytest.importorskip("CoolProp")
    problem = _problem()
    problem.components.add_process_mvr(
        "Evaporator vapour", mvr_stage_pressure_ratio=1.5
    )

    problem.load(_problem_payload())

    assert problem.process_components == {}


def test_workspace_option_and_multiplier_changes_keep_components():
    pytest.importorskip("CoolProp")
    workspace = PinchWorkspace()
    workspace.load(_problem_payload(), case_name="base")
    component = workspace.case("base").components.add_process_mvr(
        "Evaporator vapour", mvr_stage_pressure_ratio=1.5
    )

    workspace.update_options({"ENV_TEMPERATURE": 30.0}, case_name="base")
    workspace.set_dt_cont_multiplier(2.0, case_name="base")

    problem = workspace.case("base")
    assert problem.process_components == {"mvr_1": component}
    assert (
        component.stream_records[0].original_memberships[0].zone is problem._master_zone
    )
