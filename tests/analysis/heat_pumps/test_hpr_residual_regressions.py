"""Regression and generated oracles for HPR load and residual accounting."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from OpenPinch.analysis.heat_pumps.common.postprocessing import (
    _get_hpr_residual_load_profiles,
    _get_hpr_residual_utility_net_profile,
)
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.enums import ProblemTableLabel as L
from OpenPinch.domain.problem_table import ProblemTable


def test_residual_profile_is_stored_after_pocket_grid_changes():
    pt = ProblemTable(
        {L.T: [200.0, 150.0, 100.0, 50.0], L.H_NET: [100.0, 150.0, 0.0, 50.0]}
    )
    temperatures, net = _get_hpr_residual_utility_net_profile(
        T_vals=pt[L.T], residual_net=pt[L.H_NET]
    )
    assert len(temperatures) > 4
    _get_hpr_residual_load_profiles(
        pt=pt,
        T_vals=temperatures,
        residual_net=net,
        is_direct=True,
        is_heat_pumping=True,
    )
    assert np.isfinite(pt[L.H_NET_HOT_AFTR_HP]).all()
    assert np.isfinite(pt[L.H_NET_COLD_AFTR_HP]).all()


@pytest.mark.parametrize("fraction", [-0.01, 1.01, float("inf"), float("nan")])
def test_fraction_rejects_invalid_service_selection(fraction):
    with pytest.raises((ValueError, TypeError)):
        Configuration(options={"HPR_LOAD_FRACTION": fraction})


@st.composite
def residual_cascades(draw):
    heating = draw(st.floats(min_value=1, max_value=10000, allow_nan=False))
    cooling = draw(st.floats(min_value=1, max_value=10000, allow_nan=False))
    offset = draw(st.floats(min_value=-1000, max_value=1000, allow_nan=False))
    return np.array([200.0, 100.0, 20.0]), np.array([heating, 0.0, cooling]), offset


@given(residual_cascades())
def test_residual_rebase_matches_offset_independent_oracle(case):
    temperature, net, offset = case
    t, actual = _get_hpr_residual_utility_net_profile(
        T_vals=temperature, residual_net=net + offset
    )
    np.testing.assert_allclose(t, temperature)
    np.testing.assert_allclose(actual, net, atol=1e-9)
    _, repeated = _get_hpr_residual_utility_net_profile(T_vals=t, residual_net=actual)
    np.testing.assert_allclose(repeated, actual)


@given(st.floats(min_value=0, max_value=0.95), st.floats(min_value=0, max_value=1))
def test_heat_pump_ambient_sink_cannot_expand_selected_load(ambient, fraction):
    from types import SimpleNamespace

    from OpenPinch.analysis.heat_pumps.common.layout import HPRoptVectorLayout
    from OpenPinch.analysis.heat_pumps.targeting.parallel_carnot import (
        _parse_parallel_carnot_hp_state_variables,
    )

    args = SimpleNamespace(
        n_cond=1,
        n_evap=1,
        is_heat_pumping=True,
        Q_heat_max=100.0,
        Q_cool_max=200.0,
        T_cold=np.array([150.0, 100.0]),
        T_hot=np.array([80.0, 20.0]),
        H_cold=np.array([100.0, 0.0]),
        H_hot=np.array([0.0, -200.0]),
        z_amb_cold=np.array([1.0, 0.0]),
        z_amb_hot=np.array([0.0, -1.0]),
    )
    layout = HPRoptVectorLayout(n_cond=1, n_evap=1, n_heat_base=1, n_heat_split=1)
    point = layout.pack(
        x_amb=ambient,
        x_cond=[0.0],
        x_evap=[0.0],
        x_heat_base=[fraction],
        x_heat_split=[1.0],
    )
    state = _parse_parallel_carnot_hp_state_variables(point, args)
    assert state.Q_heat_base == pytest.approx(100.0 * fraction)


@given(st.floats(min_value=0, max_value=10000))
def test_unused_low_grade_heat_has_no_heat_pump_feasibility_penalty(cold):
    from types import SimpleNamespace

    from OpenPinch.analysis.heat_pumps.optimisation_adapter import build_hpr_accounting

    args = SimpleNamespace(
        is_heat_pumping=True,
        Q_hpr_target=100.0,
        heat_to_power_ratio=1.0,
        cold_to_power_ratio=0.0,
        eta_penalty=0.001,
        rho_penalty=10.0,
    )
    _, _, penalty, objective = build_hpr_accounting(
        work=10.0,
        Q_ext_heat=20.0,
        Q_ext_cold=cold,
        args=args,
        penalise_external_cold_when_refrigerating=True,
    )
    assert penalty == 0.0
    assert objective == pytest.approx(0.3)
