"""The two Varbanov models each use one size band and warn outside it."""

import pytest

from OpenPinch.analysis.power import steam_turbine as turbine_mod
from OpenPinch.analysis.power.steam_turbine import MultiStageSteamTurbine
from OpenPinch.analysis.thermodynamics.water import Tsat_p
from OpenPinch.domain.enums import TurbineModel

SMALL = TurbineModel.VARBANOV_SMALL.value
LARGE = TurbineModel.VARBANOV_LARGE.value

# Varbanov et al. (2004) coefficients as the code labels them, repeated here so a
# change to the table or to which band a model reads fails a test.
COEFFICIENTS = {
    (SMALL, "BPST"): (0, 0.00108, 1.097, 0.00172),
    (SMALL, "CT"): (0, 0.000662, 1.191, 0.000759),
    (LARGE, "BPST"): (0, 0.00423, 1.155, 0.000538),
    (LARGE, "CT"): (-0.463, 0.00353, 1.22, 0.000148),
}
P_IN, P_OUT, H_IN, DH_IS = 90.0, 1.0, 3000.0, 1000.0


def _expected_work(model, turbine_type, mass_flow):
    c0, c1, c2, c3 = COEFFICIENTS[model, turbine_type]
    dt_sat = Tsat_p(P_IN) - Tsat_p(P_OUT)
    a = (c0 + c1 * dt_sat) * 1000
    b = c2 + c3 * dt_sat
    return (DH_IS * mass_flow - a) / b


def _predict(model, mass_flow, h_sat):
    return turbine_mod._predict_stage_work(
        model=model,
        pressure_in=P_IN,
        enthalpy_in=H_IN,
        pressure_out=P_OUT,
        saturation_enthalpy=h_sat,
        mass_flow=mass_flow,
        mass_flow_max=mass_flow,
        dh_isentropic=DH_IS,
        mech_eff=1.0,
        min_eff=0.1,
    )


@pytest.mark.parametrize("model", [SMALL, LARGE])
@pytest.mark.parametrize("mass_flow", [1.0, 20.0], ids=["below_2MW", "above_2MW"])
def test_a_model_uses_its_own_band_at_any_size(model, mass_flow):
    # A dry exit keeps the turbine a back-pressure one. The two flows give about
    # 0.5 MW and 13 MW, so neither model may switch band on size.
    work = _predict(model, mass_flow, h_sat=0.0)

    assert work == pytest.approx(_expected_work(model, "BPST", mass_flow), rel=1e-12)


@pytest.mark.parametrize("model", [SMALL, LARGE])
def test_a_wet_exit_switches_type_within_the_band(model):
    work = _predict(model, 5.0, h_sat=H_IN)

    assert work == pytest.approx(_expected_work(model, "CT", 5.0), rel=1e-12)


@pytest.mark.parametrize(
    ("model", "work", "warns"),
    [
        (SMALL, 2000.0, False),
        (SMALL, 2000.5, True),
        (LARGE, 2000.0, True),
        (LARGE, 2000.5, False),
    ],
)
def test_band_limit_is_exclusive_for_large_and_inclusive_for_small(model, work, warns):
    messages = turbine_mod._varbanov_band_diagnostics(model, [work])

    assert bool(messages) is warns


def test_a_stage_doing_no_work_is_not_out_of_range():
    assert turbine_mod._varbanov_band_diagnostics(LARGE, [0.0]) == []


def test_other_models_have_no_band():
    assert (
        turbine_mod._varbanov_band_diagnostics(
            TurbineModel.SUN_SMITH.value, [1.0, 5000.0]
        )
        == []
    )


def test_the_message_names_model_work_band_and_the_other_method():
    message = turbine_mod._varbanov_band_diagnostics(SMALL, [100.0, 2345.6])[0]

    assert SMALL in message
    assert "stage 2" in message
    assert "2346 kW" in message
    assert "stage 1" not in message
    assert "at or below 2000 kW" in message
    assert "varbanov_large_turbine_cogeneration" in message

    message = turbine_mod._varbanov_band_diagnostics(LARGE, [1500.0])[0]

    assert LARGE in message
    assert "1500 kW" in message
    assert "above 2000 kW" in message
    assert "varbanov_small_turbine_cogeneration" in message


def test_all_out_of_range_stages_share_one_message_per_solve():
    # Two extraction levels, each with a duty that puts its stage far above 2 MW.
    _, details = MultiStageSteamTurbine().solve(
        [180.0, 120.0],
        [20_000.0, 20_000.0],
        mode="above_pinch",
        T_in=450.0,
        P_in=90.0,
        model=SMALL,
    )

    assert len(details["diagnostics"]) == 1
    assert "stage 1" in details["diagnostics"][0]
    assert "stage 2" in details["diagnostics"][0]
    assert details["converged"] is True


def test_the_below_pinch_solve_reports_the_band_too():
    _, details = MultiStageSteamTurbine().solve(
        [180.0],
        [50_000.0],
        mode="below_pinch",
        T_sink=40.0,
        model=SMALL,
    )

    assert len(details["diagnostics"]) == 1
    assert "at or below 2000 kW" in details["diagnostics"][0]


@pytest.mark.parametrize(
    ("model", "work", "other"),
    [
        (TurbineModel.VARBANOV_SMALL.value, 2312.0, "varbanov_large_turbine"),
        (TurbineModel.VARBANOV_LARGE.value, 1703.0, "varbanov_small_turbine"),
    ],
)
def test_out_of_range_message_does_not_promise_the_other_model_fits(model, work, other):
    [message] = turbine_mod._varbanov_band_diagnostics(model, [work])

    assert other in message
    assert "may also be out of range" in message
