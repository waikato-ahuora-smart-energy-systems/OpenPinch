"""Direct MVR stages must not cross into liquid without the source declaring it."""

from __future__ import annotations

import pickle

import pytest
from CoolProp.CoolProp import PropsSI

from OpenPinch.analysis.heat_pumps.direct_mvr.execution import (
    solve_direct_gas_mvr_stream,
)
from OpenPinch.analysis.heat_pumps.direct_mvr.models import (
    DirectGasMVRPhaseError,
    DirectGasMVRSettings,
    DirectGasMVRStageError,
)
from OpenPinch.application.commands import run_command
from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain.stream import Stream

P_SUPPLY_KPA = 101.325
PRESSURE_RATIO = 1.5
# Saturation temperature (degC) at the first stage outlet, 151.99 kPa.
T_SAT_STAGE_1 = (
    PropsSI("T", "P", P_SUPPLY_KPA * PRESSURE_RATIO * 1e3, "Q", 1.0, "Water") - 273.15
)
P_SAT_100C_KPA = PropsSI("P", "T", 373.15, "Q", 1.0, "Water") / 1e3


def _steam(t_target, *, t_supply=150.0, phase="gas", **overrides):
    return Stream(
        name="Steam",
        supply_temperature=t_supply,
        target_temperature=t_target,
        supply_pressure=overrides.pop("supply_pressure", P_SUPPLY_KPA),
        heat_flow=100.0,
        delta_t_contribution=0.0,
        heat_transfer_coefficient=1.0,
        fluid_name="Water",
        fluid_phase=phase,
        **overrides,
    )


def _saturated_steam():
    return _steam(
        99.5,
        t_supply=100.0,
        phase="vapour",
        supply_pressure=P_SAT_100C_KPA,
    )


def _settings(**overrides):
    return DirectGasMVRSettings(
        **{"mvr_stage_pressure_ratio": PRESSURE_RATIO, **overrides}
    )


def _problem(*, t_target, t_supply=150.0, options=None) -> PinchProblem:
    return PinchProblem(
        {
            "streams": [
                {
                    "zone": "Site/P",
                    "name": "Steam",
                    "t_supply": t_supply,
                    "t_target": t_target,
                    "p_supply": P_SUPPLY_KPA,
                    "heat_flow": 100.0,
                    "dt_cont": 0.0,
                    "htc": 1.0,
                    "fluid_name": "Water",
                    "fluid_phase": "gas",
                },
                {
                    "zone": "Site/P",
                    "name": "Cold",
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
                "children": [{"name": "P", "type": "Process Zone"}],
            },
            "options": options or {},
        },
        project_name="Site",
    )


@pytest.mark.parametrize("liquid_injection", [True, False])
def test_gas_source_target_below_stage_saturation_raises(liquid_injection):
    with pytest.raises(DirectGasMVRPhaseError) as caught:
        solve_direct_gas_mvr_stream(
            _steam(T_SAT_STAGE_1 - 0.1),
            settings=_settings(liquid_injection=liquid_injection),
        )

    error = caught.value
    assert error.reason_code == "phase.target_below_saturation"
    assert error.stage_index == 1
    message = str(error)
    assert "'Steam'" in message
    assert "stage 1" in message
    assert f"{T_SAT_STAGE_1 - 0.1:.2f} degC" in message
    assert f"{T_SAT_STAGE_1:.2f} degC" in message
    assert f"{P_SUPPLY_KPA * PRESSURE_RATIO:.2f} kPa" in message


def test_gas_source_target_just_above_stage_saturation_still_solves():
    solved = solve_direct_gas_mvr_stream(
        _steam(T_SAT_STAGE_1 + 0.1),
        settings=_settings(),
    )

    stage = solved.stage_results[0]
    assert stage.liquid_injection_applied is False
    # Vapour throughout: the heat released is the source duty plus shaft work, plus
    # the small drop in vapour enthalpy from the supply to the outlet pressure.
    t_target_kelvin = T_SAT_STAGE_1 + 0.1 + 273.15
    h_target_at_supply_pressure = (
        PropsSI("H", "T", t_target_kelvin, "P", P_SUPPLY_KPA * 1e3, "Water") / 1e3
    )
    pressure_term = stage.source_mass_flow * (
        h_target_at_supply_pressure - stage.h_target
    )
    assert stage.heat_flow == pytest.approx(
        100.0 + stage.work * 0.95 + pressure_term, rel=1e-6
    )


def test_gas_source_names_the_later_stage_that_crosses_saturation():
    t_sat_stage_2 = (
        PropsSI("T", "P", P_SUPPLY_KPA * PRESSURE_RATIO**2 * 1e3, "Q", 1.0, "Water")
        - 273.15
    )

    with pytest.raises(DirectGasMVRPhaseError) as caught:
        solve_direct_gas_mvr_stream(
            _steam(t_sat_stage_2 - 1.0, t_supply=170.0),
            settings=_settings(n_stages=2),
        )

    assert caught.value.reason_code == "phase.target_below_saturation"
    assert caught.value.stage_index == 2
    assert "stage 2" in str(caught.value)


def test_saturated_vapour_source_still_condenses_in_one_stage():
    solved = solve_direct_gas_mvr_stream(
        _saturated_steam(),
        settings=_settings(mvr_stage_pressure_ratio=1.2),
    )

    stage = solved.stage_results[0]
    assert stage.heat_flow == pytest.approx(100.0 + stage.work * 0.95, rel=1e-3)


@pytest.mark.parametrize(
    "settings",
    [
        {"mvr_stage_pressure_ratio": 1.2},
        {"mvr_stage_pressure_ratio": None, "mvr_stage_t_lift": 20.0},
    ],
    ids=["pressure_ratio", "t_lift"],
)
def test_saturated_vapour_source_rejects_a_stage_starting_from_liquid(settings):
    with pytest.raises(DirectGasMVRPhaseError) as caught:
        solve_direct_gas_mvr_stream(
            _saturated_steam(),
            settings=_settings(n_stages=2, **settings),
        )

    error = caught.value
    assert error.reason_code == "phase.liquid_stage_inlet"
    assert error.stage_index == 2
    message = str(error)
    assert "stage 2" in message
    assert "99.50 degC" in message
    assert "liquid" in message


def test_source_declaring_a_liquid_target_enthalpy_may_condense():
    solved = solve_direct_gas_mvr_stream(
        _steam(
            105.0,
            t_supply=120.0,
            supply_enthalpy=2716.0,
            target_enthalpy=420.0,
        ),
        settings=_settings(),
    )

    stage = solved.stage_results[0]
    assert stage.h_target < 500.0
    assert stage.heat_flow == pytest.approx(100.0 + stage.work * 0.95, rel=0.02)


def test_air_is_not_rejected_above_its_saturation_range():
    stream = _steam(80.0, t_supply=120.0)
    stream.fluid_name = "Air"

    solved = solve_direct_gas_mvr_stream(stream, settings=_settings(n_stages=2))

    assert len(solved.stage_results) == 2


def test_phase_error_survives_pickling_for_process_workers():
    error = DirectGasMVRPhaseError(
        reason_code="phase.target_below_saturation",
        source_stream="Steam",
        period_index=1,
        stage_index=2,
        fluid="Water",
        t_target=100.0,
        t_saturation=104.2,
        pressure_kpa=117.6,
    )

    restored = pickle.loads(pickle.dumps(error))

    assert isinstance(restored, DirectGasMVRStageError)
    assert str(restored) == str(error)
    assert restored.pressure_kpa == pytest.approx(117.6)


def test_phase_error_bounds_a_long_stream_name():
    error = DirectGasMVRPhaseError(
        reason_code="phase.liquid_stage_inlet",
        source_stream="S" * 500,
        period_index=0,
        stage_index=2,
        fluid="Water",
        t_target=100.0,
        t_saturation=104.2,
        pressure_kpa=117.6,
    )

    assert len(str(error)) < 500


def test_add_process_mvr_raises_and_leaves_the_problem_unchanged():
    problem = _problem(t_target=T_SAT_STAGE_1 - 0.1)
    before = problem.to_problem_json()

    with pytest.raises(DirectGasMVRPhaseError, match="stage 1"):
        problem.components.add_process_mvr(
            "Steam", mvr_stage_pressure_ratio=PRESSURE_RATIO
        )

    assert problem.process_components == {}
    assert problem.to_problem_json() == before
    streams = problem.master_zone.get_subzone("P").hot_streams
    assert all(stream.is_active for stream in streams)


def test_add_process_mvr_names_the_period_that_crosses_saturation():
    problem = _problem(
        t_target={
            "values": [T_SAT_STAGE_1 + 0.5, T_SAT_STAGE_1 - 0.5],
            "unit": "degC",
        },
        t_supply={"values": [150.0, 150.0], "unit": "degC"},
        options={"PROBLEM_PERIOD_IDS": ["0", "peak"]},
    )

    with pytest.raises(DirectGasMVRPhaseError) as caught:
        problem.components.add_process_mvr(
            "Steam", mvr_stage_pressure_ratio=PRESSURE_RATIO
        )

    assert caught.value.period_index == 1
    assert "period 1" in str(caught.value)


def test_replay_that_crosses_saturation_raises_and_keeps_the_prior_state():
    probe = solve_direct_gas_mvr_stream(
        _steam(120.0),
        settings=DirectGasMVRSettings(mvr_stage_t_lift=20.0, eta_mvr_comp=0.7),
    )
    p_out_kpa = probe.stage_results[0].p_out
    t_sat = PropsSI("T", "P", p_out_kpa * 1e3, "Q", 1.0, "Water") - 273.15
    problem = _problem(t_target=t_sat + 0.05)
    component = problem.components.add_process_mvr("Steam", mvr_stage_t_lift=20.0)
    before = component.stage_results_by_period["0"][0].p_out

    # A better compressor needs more pressure for the same discharge temperature.
    with pytest.raises(DirectGasMVRPhaseError, match="stage 1"):
        problem.update_options({"PROCESS_MVR_ETA_COMP": 0.95})

    assert problem.process_components["mvr_1"] is component
    assert component.stage_results_by_period["0"][0].p_out == pytest.approx(before)


def test_saved_case_that_now_crosses_saturation_fails_to_load_with_the_reason():
    saved = _problem(t_target=T_SAT_STAGE_1 + 0.1).to_problem_json()
    saved["streams"][0]["t_target"] = T_SAT_STAGE_1 - 0.1
    saved["process_components"] = [
        {
            "type": "process_mvr",
            "id": "mvr_1",
            "source_streams": [{"zone": "P", "name": "Steam"}],
            "n_stages": 1,
            "liquid_injection": True,
            "mvr_stage_pressure_ratio": PRESSURE_RATIO,
            "active": True,
        }
    ]

    with pytest.raises(DirectGasMVRPhaseError, match="stage 1"):
        PinchProblem(saved)


def test_command_layer_reports_the_phase_error_message():
    session = run_command(
        {
            "command": "open",
            "problem": {
                "data": _problem(t_target=T_SAT_STAGE_1 - 0.1).to_problem_json()
            },
        }
    )["session"]

    with pytest.raises(ValueError, match="below the saturation temperature"):
        run_command(
            {
                "command": "components.add_process_mvr",
                "session": session,
                "arguments": {
                    "source_streams": "Steam",
                    "mvr_stage_pressure_ratio": PRESSURE_RATIO,
                },
            }
        )
