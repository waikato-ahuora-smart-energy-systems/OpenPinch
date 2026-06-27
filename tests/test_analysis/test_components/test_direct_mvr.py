from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from CoolProp.CoolProp import PropsSI

from OpenPinch.classes.stream import Stream
from OpenPinch.classes.value import Value
from OpenPinch.services.components.direct_mvr import (
    DirectGasMVROutputUnits,
    DirectGasMVRSettings,
    DirectGasMVRStageResult,
    solve_direct_gas_mvr_stream,
)
from OpenPinch.services.components.direct_mvr import (
    direct_gas_mvr as direct_mvr_helpers,
)

DIRECT_MVR_STAGE_CASES = (
    Path(__file__).parents[2] / "fixtures" / "process_mvr_edge_cases.json"
)


def _gas_stream(name="Gas", *, fluid="Air", p=101.325, heat_flow=100.0):
    return Stream(
        name=name,
        t_supply=120.0,
        t_target=80.0,
        p_supply=p,
        heat_flow=heat_flow,
        dt_cont=0.0,
        htc=1.0,
        fluid_name=fluid,
        fluid_phase="gas",
    )


def _water_vapour_stream(name="Water vapour", *, heat_flow=100.0):
    p_sat_kpa = Value(
        PropsSI("P", "T", Value(100.0, "degC").to("K").value, "Q", 1.0, "Water"),
        "Pa",
    ).to("kPa")
    return Stream(
        name=name,
        t_supply=100.0,
        t_target=99.5,
        p_supply=p_sat_kpa,
        heat_flow=heat_flow,
        dt_cont=0.0,
        htc=1.0,
        fluid_name="Water",
        fluid_phase="vapour",
    )


class CustomUnitStream(Stream):
    _VALUE_UNITS = {
        **Stream._VALUE_UNITS,
        "_t_supply": "K",
        "_t_target": "K",
        "_p_supply": "bar",
        "_p_target": "bar",
        "_h_supply": "J/kg",
        "_h_target": "J/kg",
        "_heat_flow": "MW",
    }


def _stage_result_from_fixture(name: str, **overrides) -> DirectGasMVRStageResult:
    payload = json.loads(DIRECT_MVR_STAGE_CASES.read_text())
    stage_payload = {**payload[name], **overrides}
    profile = np.asarray(payload["stage_profile"], dtype=float)
    return DirectGasMVRStageResult(
        **stage_payload,
        th_curve=profile.copy(),
        linearised_profile=profile.copy(),
    )


def test_direct_mvr_solver_builds_multistage_replacement_streams():
    pytest.importorskip("CoolProp")
    source = _gas_stream()
    settings = DirectGasMVRSettings(
        n_stages=2,
        mvr_stage_t_lift=10.0,
        liquid_injection=False,
        eta_mvr_comp=0.75,
        eta_motor=0.95,
    )

    solved = solve_direct_gas_mvr_stream(source, settings=settings)

    assert len(solved.stage_results) == 2
    assert len(solved.replacement_streams) >= 2
    assert sum(stage.work for stage in solved.stage_results) > 0.0
    assert sum(stage.heat_flow for stage in solved.stage_results) > 0.0
    for replacement in solved.replacement_streams:
        assert replacement.fluid_phase == "gas"
        assert replacement.t_supply.unit == "degC"
        assert replacement.t_target.unit == "degC"
        assert replacement.p_supply.unit == "kPa"
        assert replacement.p_target.unit == "kPa"
        assert replacement.heat_flow.unit == "kW"
        assert float(replacement.p_supply) > float(source.p_supply)
    assert min(float(stream.t_target) for stream in solved.replacement_streams) == (
        pytest.approx(float(source.t_target), abs=0.05)
    )
    assert sum(float(stream.heat_flow) for stream in solved.replacement_streams) == (
        pytest.approx(sum(stage.heat_flow for stage in solved.stage_results))
    )


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (DirectGasMVRSettings(n_stages=0), "positive integer"),
        (DirectGasMVRSettings(n_stages=1.5), "positive integer"),
        (DirectGasMVRSettings(n_stages=True), "positive integer"),
        (DirectGasMVRSettings(mvr_stage_t_lift=0.0), "mvr_stage_t_lift"),
        (
            DirectGasMVRSettings(mvr_stage_pressure_ratio=1.0),
            "mvr_stage_pressure_ratio",
        ),
        (DirectGasMVRSettings(eta_mvr_comp=0.0), "eta_mvr_comp"),
        (DirectGasMVRSettings(eta_motor=1.01), "eta_motor"),
    ],
)
def test_direct_mvr_solver_rejects_invalid_settings(settings, message):
    pytest.importorskip("CoolProp")

    with pytest.raises(ValueError, match=message):
        solve_direct_gas_mvr_stream(_gas_stream(), settings=settings)


def test_direct_mvr_solver_can_use_pressure_ratio_target():
    pytest.importorskip("CoolProp")
    source = _gas_stream(p=100.0)

    solved = solve_direct_gas_mvr_stream(
        source,
        settings=DirectGasMVRSettings(mvr_stage_pressure_ratio=1.25),
    )

    stage = solved.stage_results[0]
    assert stage.p_out / stage.p_in == pytest.approx(1.25)
    assert float(solved.replacement_streams[0].p_supply) == pytest.approx(stage.p_out)


def test_direct_mvr_stage_result_enthalpies_use_stream_input_units():
    pytest.importorskip("CoolProp")
    solved = solve_direct_gas_mvr_stream(
        _gas_stream(),
        settings=DirectGasMVRSettings(mvr_stage_pressure_ratio=1.2),
    )

    stage = solved.stage_results[0]

    assert 0.0 < stage.h_target < stage.h_hot_supply < 2_000.0
    assert stage.th_curve[0, 0] == pytest.approx(stage.h_hot_supply)
    assert stage.th_curve[-1, 0] == pytest.approx(stage.h_target)
    assert stage.linearised_profile[0, 0] == pytest.approx(stage.h_hot_supply)
    assert stage.linearised_profile[-1, 0] == pytest.approx(stage.h_target)


def test_direct_mvr_stage_result_units_follow_source_stream_units():
    pytest.importorskip("CoolProp")
    source = CustomUnitStream(
        name="Custom units",
        t_supply=Value(393.15, "K"),
        t_target=Value(353.15, "K"),
        p_supply=Value(1.01325, "bar"),
        h_supply=Value(420000.0, "J/kg"),
        h_target=Value(380000.0, "J/kg"),
        heat_flow=Value(0.1, "MW"),
        dt_cont=0.0,
        htc=1.0,
        fluid_name="Air",
        fluid_phase="gas",
    )

    solved = solve_direct_gas_mvr_stream(
        source,
        settings=DirectGasMVRSettings(mvr_stage_pressure_ratio=1.2),
    )

    stage = solved.stage_results[0]
    assert stage.temperature_unit == "K"
    assert stage.pressure_unit == "bar"
    assert stage.enthalpy_unit == "J/kg"
    assert stage.heat_flow_unit == "MW"
    assert stage.p_in == pytest.approx(1.01325)
    assert stage.p_out / stage.p_in == pytest.approx(1.2)
    assert stage.t_in == pytest.approx(393.15)
    assert stage.t_target == pytest.approx(353.15)
    assert stage.h_hot_supply > stage.h_target > 0.0
    assert stage.heat_flow > 0.0
    assert stage.work > 0.0
    assert stage.th_curve[0, 0] == pytest.approx(stage.h_hot_supply)
    assert stage.th_curve[0, 1] == pytest.approx(stage.t_hot_supply)
    assert stage.linearised_profile[-1, 0] == pytest.approx(stage.h_target)


def test_direct_mvr_solver_normalises_source_enthalpy_units():
    pytest.importorskip("CoolProp")
    settings = DirectGasMVRSettings(
        mvr_stage_pressure_ratio=1.2,
        liquid_injection=False,
    )
    kj_source = _gas_stream()
    kj_source.h_supply = Value(420.0, "kJ/kg")
    kj_source.h_target = Value(380.0, "kJ/kg")
    j_source = _gas_stream()
    j_source.h_supply = Value(420000.0, "J/kg")
    j_source.h_target = Value(380000.0, "J/kg")

    kj_solved = solve_direct_gas_mvr_stream(kj_source, settings=settings)
    j_solved = solve_direct_gas_mvr_stream(j_source, settings=settings)

    assert j_solved.stage_results[0].work == pytest.approx(
        kj_solved.stage_results[0].work
    )
    assert sum(float(stream.heat_flow) for stream in j_solved.replacement_streams) == (
        pytest.approx(
            sum(float(stream.heat_flow) for stream in kj_solved.replacement_streams)
        )
    )


def test_direct_mvr_solver_rejects_conflicting_compression_targets():
    pytest.importorskip("CoolProp")

    with pytest.raises(
        ValueError,
        match="either mvr_stage_t_lift or mvr_stage_pressure_ratio",
    ):
        solve_direct_gas_mvr_stream(
            _gas_stream(),
            settings=DirectGasMVRSettings(
                mvr_stage_t_lift=10.0,
                mvr_stage_pressure_ratio=1.2,
            ),
        )


@pytest.mark.parametrize(
    ("stream", "message"),
    [
        (_gas_stream(p=None), "p_supply"),
        (_gas_stream(heat_flow=0.0), "positive duty"),
        (
            Stream(
                name="NoCooling",
                t_supply=80.0,
                t_target=120.0,
                p_supply=101.325,
                heat_flow=100.0,
                fluid_name="Air",
                fluid_phase="gas",
            ),
            "cool from supply to target",
        ),
    ],
)
def test_direct_mvr_solver_rejects_missing_or_invalid_stream_data(stream, message):
    pytest.importorskip("CoolProp")

    with pytest.raises(ValueError, match=message):
        solve_direct_gas_mvr_stream(stream, settings=DirectGasMVRSettings())


def test_direct_mvr_liquid_injection_toggle_is_recorded():
    pytest.importorskip("CoolProp")
    dry = solve_direct_gas_mvr_stream(
        _gas_stream(),
        settings=DirectGasMVRSettings(
            liquid_injection=False,
            mvr_stage_t_lift=10.0,
        ),
    )
    injected = solve_direct_gas_mvr_stream(
        _gas_stream(),
        settings=DirectGasMVRSettings(
            liquid_injection=True,
            mvr_stage_t_lift=10.0,
        ),
    )

    assert dry.stage_results[0].liquid_injection_applied is False
    assert injected.stage_results[0].liquid_injection_applied in {False, True}


def test_direct_mvr_liquid_injection_uses_post_injection_mass_basis():
    pytest.importorskip("CoolProp")
    settings = DirectGasMVRSettings(
        liquid_injection=False,
        mvr_stage_pressure_ratio=1.2,
    )
    dry = solve_direct_gas_mvr_stream(_water_vapour_stream(), settings=settings)
    injected = solve_direct_gas_mvr_stream(
        _water_vapour_stream(),
        settings=DirectGasMVRSettings(
            liquid_injection=True,
            mvr_stage_pressure_ratio=1.2,
        ),
    )

    dry_stage = dry.stage_results[0]
    injected_stage = injected.stage_results[0]

    assert injected_stage.liquid_injection_applied is True
    assert injected_stage.source_mass_flow == pytest.approx(dry_stage.source_mass_flow)
    assert injected_stage.hot_mass_flow > injected_stage.source_mass_flow
    assert injected_stage.liquid_injection_ratio > 0.0
    assert injected_stage.work == pytest.approx(dry_stage.work)
    assert injected_stage.heat_flow == pytest.approx(dry_stage.heat_flow)
    assert sum(float(stream.heat_flow) for stream in injected.replacement_streams) == (
        pytest.approx(injected_stage.heat_flow)
    )


def test_direct_mvr_liquid_injection_mixer_conserves_enthalpy():
    pytest.importorskip("CoolProp")
    dry = solve_direct_gas_mvr_stream(
        _water_vapour_stream(),
        settings=DirectGasMVRSettings(
            liquid_injection=False,
            mvr_stage_pressure_ratio=1.2,
        ),
    )
    injected = solve_direct_gas_mvr_stream(
        _water_vapour_stream(),
        settings=DirectGasMVRSettings(
            liquid_injection=True,
            mvr_stage_pressure_ratio=1.2,
        ),
    )

    dry_stage = dry.stage_results[0]
    injected_stage = injected.stage_results[0]
    ratio = injected_stage.liquid_injection_ratio

    assert ratio > 0.0
    assert dry_stage.h_hot_supply + ratio * injected_stage.h_target == pytest.approx(
        (1.0 + ratio) * injected_stage.h_hot_supply,
        rel=1e-8,
    )
    assert injected_stage.q_liquid_injection == pytest.approx(
        ratio
        * injected_stage.source_mass_flow
        * (injected_stage.h_hot_supply - injected_stage.h_target),
        rel=1e-8,
    )


def test_direct_mvr_solver_rejects_enthalpy_and_mass_flow_edge_cases():
    pytest.importorskip("CoolProp")
    no_drop = _gas_stream()
    no_drop.h_supply = Value(100.0, "kJ/kg")
    no_drop.h_target = Value(100.0, "kJ/kg")
    tiny_mass_flow = _gas_stream(heat_flow=1e-5)
    tiny_mass_flow.h_supply = Value(1e12, "kJ/kg")
    tiny_mass_flow.h_target = Value(0.0, "kJ/kg")

    with pytest.raises(ValueError, match="no positive enthalpy drop"):
        solve_direct_gas_mvr_stream(no_drop, settings=DirectGasMVRSettings())
    with pytest.raises(ValueError, match="non-positive mass flow"):
        solve_direct_gas_mvr_stream(tiny_mass_flow, settings=DirectGasMVRSettings())


def test_direct_mvr_value_stage_and_stage_count_helpers_cover_scalar_edges():
    positive_stage = _stage_result_from_fixture("positive_stage", hot_mass_flow=0.0)
    zero_delta_stage = _stage_result_from_fixture("zero_delta_stage", hot_mass_flow=0.0)
    explicit_mass_stage = _stage_result_from_fixture(
        "positive_stage", hot_mass_flow=2.5
    )

    with pytest.raises(ValueError, match="positive integer"):
        direct_mvr_helpers.coerce_positive_mvr_stage_count("not-a-number")

    assert direct_mvr_helpers._stage_mass_flow(explicit_mass_stage) == pytest.approx(
        2.5
    )
    assert direct_mvr_helpers._stage_mass_flow(zero_delta_stage) == 0.0
    assert direct_mvr_helpers._stage_mass_flow(positive_stage) == pytest.approx(0.5)
    assert direct_mvr_helpers._stage_segment_heat_flow(
        positive_stage,
        300.0,
        200.0,
    ).value == pytest.approx(50.0)
    assert direct_mvr_helpers._value(None, 0) is None
    assert direct_mvr_helpers._value(12.5, 99) == pytest.approx(12.5)
    assert direct_mvr_helpers._value(Value([1.0, 2.0], "kW"), 1, unit="W") == (
        pytest.approx(2000.0)
    )
    assert direct_mvr_helpers._stage_pressure_to_pa(1.2, "bar") == pytest.approx(
        120000.0
    )


def test_direct_mvr_state_property_fallback_and_failure_paths(monkeypatch):
    pytest.importorskip("CoolProp")

    def saturated_property(*args):
        if args[1] == "T" and args[3] == "P":
            raise ValueError("ambiguous saturation state")
        if args[0] == "T":
            return Value(100.0, "degC").to("K").value
        return 1234.0

    monkeypatch.setattr(direct_mvr_helpers, "PropsSI", saturated_property)

    assert direct_mvr_helpers._state_property_at_temperature_pressure(
        fluid="Water",
        output="H",
        t_c=100.0,
        p_kpa=101.325,
        saturated_quality=1.0,
    ) == pytest.approx(1234.0)

    def distant_saturation_temperature(*args):
        if args[1] == "T" and args[3] == "P":
            raise ValueError("ambiguous saturation state")
        if args[0] == "T":
            return Value(50.0, "degC").to("K").value
        return 1234.0

    monkeypatch.setattr(direct_mvr_helpers, "PropsSI", distant_saturation_temperature)
    with pytest.raises(ValueError, match="ambiguous saturation state"):
        direct_mvr_helpers._state_property_at_temperature_pressure(
            fluid="Water",
            output="H",
            t_c=100.0,
            p_kpa=101.325,
            saturated_quality=1.0,
        )

    def failed_saturation_lookup(*args):
        if args[1] == "T" and args[3] == "P":
            raise ValueError("ambiguous saturation state")
        raise RuntimeError("no saturation data")

    monkeypatch.setattr(direct_mvr_helpers, "PropsSI", failed_saturation_lookup)
    with pytest.raises(RuntimeError, match="no saturation data"):
        direct_mvr_helpers._state_property_at_temperature_pressure(
            fluid="Water",
            output="H",
            t_c=100.0,
            p_kpa=101.325,
            saturated_quality=1.0,
        )


def test_direct_mvr_pressure_search_and_profile_helpers_cover_failure_edges(
    monkeypatch,
):
    with pytest.raises(ValueError, match="no heat-release enthalpy range"):
        direct_mvr_helpers._build_cooling_th_curve(
            fluid="Air",
            p_out=101325.0,
            h_hot_supply=100.0,
            h_target=100.0,
        )

    monkeypatch.setattr(
        direct_mvr_helpers,
        "_actual_outlet_at_pressure",
        lambda *_args, **_kwargs: (100.0, 300.0),
    )
    with pytest.raises(ValueError, match="feasible MVR discharge pressure"):
        direct_mvr_helpers._find_pressure_for_actual_discharge(
            fluid="Air",
            p_in=101325.0,
            h_in=100.0,
            s_in=1.0,
            target_discharge_k=500.0,
            eta_comp=0.75,
        )

    monkeypatch.setattr(
        direct_mvr_helpers,
        "PropsSI",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("no props")),
    )
    enthalpy_values = direct_mvr_helpers._profile_enthalpy_values(
        fluid="Air",
        p_out=101325.0,
        h_hot_supply=300.0,
        h_target=100.0,
    )

    assert enthalpy_values[0] == pytest.approx(300.0)
    assert enthalpy_values[-1] == pytest.approx(100.0)
    assert len(enthalpy_values) == 31


def test_direct_mvr_liquid_injection_property_failure_falls_back_to_dry_stage(
    monkeypatch,
):
    def fake_props(output, input_1, value_1, input_2=None, value_2=None, fluid=None):
        if input_1 == "P" and input_2 == "Q":
            raise RuntimeError("saturation data unavailable")
        if output == "S":
            return 10.0
        if output == "T":
            return Value(150.0, "degC").to("K").value
        return 900.0

    monkeypatch.setattr(direct_mvr_helpers, "PropsSI", fake_props)
    monkeypatch.setattr(
        direct_mvr_helpers,
        "_actual_outlet_at_pressure",
        lambda *_args, **_kwargs: (1200.0, Value(150.0, "degC").to("K").value),
    )
    monkeypatch.setattr(
        direct_mvr_helpers,
        "_build_cooling_th_curve",
        lambda **_kwargs: np.asarray([[1200.0, 150.0], [900.0, 100.0]]),
    )
    monkeypatch.setattr(
        direct_mvr_helpers,
        "get_piecewise_data_points",
        lambda curve, **_kwargs: curve,
    )

    stage = direct_mvr_helpers._solve_compression_stage(
        fluid="FakeFluid",
        source_stream="HotGas",
        stage_index=1,
        m_dot=1.0,
        p_in=101325.0,
        t_in=120.0,
        t_target=80.0,
        compression_target=("pressure_ratio", 1.2),
        eta_comp=0.75,
        eta_motor=0.95,
        liquid_injection=True,
        dt_diff_max=0.1,
        output_units=DirectGasMVROutputUnits(),
    )

    assert stage.liquid_injection_applied is False
    assert stage.q_liquid_injection == 0.0
    assert stage.hot_mass_flow == pytest.approx(stage.source_mass_flow)
