"""Simulated (CoolProp) ORC cycles and designs."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("CoolProp")

from OpenPinch.analysis.orc.cycle import (  # noqa: E402
    CRITICAL_MARGIN,
    OrcCycleError,
    _state,
    critical_temperature,
    solve_orc_cycle,
)
from OpenPinch.analysis.orc.inputs import OrcTargetInputs  # noqa: E402
from OpenPinch.analysis.orc.profile import (  # noqa: E402
    orc_heat_source_from_gcc,
    share_at,
)
from OpenPinch.analysis.orc.simulated import (  # noqa: E402
    optimise_simulated_orc,
    simulated_orc_design,
)

T_GCC = np.array([250.0, 200.0, 150.0, 100.0, 20.0])
H_GCC = np.array([1000.0, 500.0, 0.0, 2000.0, 4000.0])


def _carnot_efficiency(T_hot: float, T_cold: float) -> float:
    return 1.0 - (T_cold + 273.15) / (T_hot + 273.15)


@pytest.mark.parametrize("fluid", ["Isopentane", "R1233zd(E)", "n-Pentane"])
def test_cycle_balances_and_stays_below_carnot(fluid):
    cycle = solve_orc_cycle(fluid, T_evap=100.0, T_cond=30.0)

    assert cycle.q_in == pytest.approx(cycle.w_net + cycle.q_out, rel=1e-9)
    assert 0.0 < cycle.eta_thermal < _carnot_efficiency(100.0, 30.0)
    # Real low-temperature ORCs reach roughly 8-14 % between 100 and 30 degC.
    assert 0.06 < cycle.eta_thermal < 0.15


def test_recuperator_raises_efficiency_of_a_dry_fluid():
    plain = solve_orc_cycle("Toluene", T_evap=140.0, T_cond=30.0, superheat=10.0)
    recuperated = solve_orc_cycle(
        "Toluene", T_evap=140.0, T_cond=30.0, superheat=10.0, dt_recuperator=10.0
    )

    assert recuperated.h2r > recuperated.h2
    assert recuperated.q_in == pytest.approx(
        recuperated.w_net + recuperated.q_out, rel=1e-9
    )
    assert recuperated.eta_thermal > plain.eta_thermal
    assert recuperated.w_net == pytest.approx(plain.w_net, rel=1e-9)


def test_evaporation_near_the_critical_point_is_rejected():
    T_crit = critical_temperature("R1234ze(E)")

    with pytest.raises(OrcCycleError, match="subcritical"):
        solve_orc_cycle("R1234ze(E)", T_evap=T_crit - 1.0, T_cond=30.0)


def test_each_thread_has_its_own_coolprop_state():
    states = {}

    def record(name):
        states[name] = (_state("Isopentane"), _state("Isopentane"))

    threads = [threading.Thread(target=record, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    (a1, a2), (b1, b2) = states[0], states[1]
    assert a1 is a2
    assert b1 is b2
    assert a1 is not b1


def _cycle_outcome(case: dict) -> tuple:
    """Everything a solve reports for ``case``, including its error."""
    try:
        cycle = solve_orc_cycle(**case)
    except OrcCycleError as exc:
        return ("error", str(exc))
    return (
        cycle.w_net,
        cycle.q_in,
        cycle.q_out,
        cycle.h3,
        cycle.T3,
        cycle.turbine_exit_quality,
    )


def _parallel_outcomes(cases: list[dict], workers: int = 8) -> list[tuple]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_cycle_outcome, cases))


@st.composite
def _cycle_case(draw) -> dict:
    fluid = draw(st.sampled_from(OrcTargetInputs().fluids))
    T_cond = draw(st.floats(min_value=20.0, max_value=50.0))
    T_evap_max = critical_temperature(fluid) - CRITICAL_MARGIN
    T_evap = draw(st.floats(min_value=T_cond + 10.0, max_value=T_evap_max))
    return {
        "fluid": fluid,
        "T_evap": T_evap,
        "T_cond": T_cond,
        "superheat": draw(st.sampled_from([0.0, 5.0, 20.0])),
        "eta_turbine": draw(st.floats(min_value=0.6, max_value=0.9)),
        "eta_pump": draw(st.floats(min_value=0.5, max_value=0.8)),
        "dt_recuperator": draw(st.sampled_from([None, 5.0, 15.0])),
    }


@settings(max_examples=40, derandomize=True, deadline=None)
@given(cases=st.lists(_cycle_case(), min_size=2, max_size=12))
def test_parallel_cycle_solves_match_serial_solves(cases):
    # Each case is solved alone first: the serial result is the oracle.
    serial = [_cycle_outcome(case) for case in cases]
    # Repeat the batch so several threads solve different fluids and
    # temperatures at once.
    assert _parallel_outcomes(cases * 4) == serial * 4


def test_parallel_isopentane_sweep_matches_serial_solves():
    # Regression: with one CoolProp state per process, threads overwrote
    # each other's state between update() and the property reads.
    cases = [
        {"fluid": "Isopentane", "T_evap": float(T), "T_cond": 30.0}
        for T in np.linspace(70.0, 150.0, 17)
    ]
    serial = [_cycle_outcome(case) for case in cases]
    for _ in range(5):
        assert _parallel_outcomes(cases) == serial


def test_evaporator_profile_runs_from_turbine_inlet_to_pump_outlet():
    cycle = solve_orc_cycle("Isopentane", T_evap=100.0, T_cond=30.0, superheat=5.0)

    T, share = cycle.evaporator_profile(0.01)

    assert np.all(np.diff(T) < 0.0)
    assert share[0] == 0.0 and share[-1] == pytest.approx(1.0)
    assert np.all(np.diff(share) >= 0.0)
    assert T[0] == pytest.approx(105.0)


@pytest.mark.parametrize("n_stages", [1, 2])
def test_simulated_designs_never_need_extra_hot_utility(n_stages):
    source = orc_heat_source_from_gcc(T_GCC, H_GCC)
    inputs = OrcTargetInputs(n_stages=n_stages)
    rng = np.random.default_rng(20260715)

    for _ in range(25):
        design = simulated_orc_design(
            rng.random(3 * n_stages), source, inputs, "Isopentane"
        )
        assert design is not None
        grid = np.linspace(source.T_min - 20.0, source.T_pinch, 400)
        residual = np.array([source.heat_at(t) for t in grid])
        for (T_profile, share), Q in zip(design.profiles, design.Q_in, strict=True):
            residual -= Q * share_at(grid, np.array(T_profile), np.array(share))
        assert residual.min() >= -1e-6
        assert max(design.T_evap_shifted) <= source.T_pinch + 1e-9


def test_simulated_orc_makes_less_power_than_its_carnot_limit():
    source = orc_heat_source_from_gcc(T_GCC, H_GCC)
    inputs = OrcTargetInputs(
        fluids=("Isopentane", "R1233zd(E)"), max_multistart=2, maximum_iterations=40
    )

    result = optimise_simulated_orc(source, inputs)
    design = result.design

    assert design.fluid in {"Isopentane", "R1233zd(E)"}
    assert result.costs.total_annualized_cost_change < 0.0
    for T_e, Q, W in zip(design.T_evap, design.Q_in, design.W_net, strict=True):
        assert W < _carnot_efficiency(T_e, inputs.T_cond) * Q
