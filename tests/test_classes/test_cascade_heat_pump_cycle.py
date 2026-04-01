"""Regression tests for the cascade heat pump cycle classes."""

import numpy as np
import pytest

pytest.importorskip("CoolProp")

from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.classes.cascade_heat_pump import CascadeHeatPumpCycle
from OpenPinch.classes.simple_heat_pump import SimpleHeatPumpCycle


def _cascade_stage_temperatures(T_evap, T_cond, dt_cascade_hx):
    """Return expected per-stage saturation temperatures for a cascade."""
    T_cond_sorted = np.sort(np.asarray(T_cond, dtype=float))[::-1]
    T_evap_sorted = np.sort(np.asarray(T_evap, dtype=float))[::-1]

    T_cond_all = np.sort(
        np.concatenate([T_cond_sorted, T_evap_sorted[:-1] + dt_cascade_hx])
    )[::-1]
    T_evap_all = np.sort(
        np.concatenate([T_cond_sorted[1:] - dt_cascade_hx, T_evap_sorted])
    )[::-1]
    return T_evap_all, T_cond_all


def _assert_stage_matches_simple(cascade, i, *, T_evap, T_cond, **kwargs):
    """Assert that stage matches simple for this test module."""
    ref = SimpleHeatPumpCycle()
    ref.solve(T_evap=T_evap, T_cond=T_cond, **kwargs)

    stage = cascade._subcycles[i]
    assert np.isclose(stage.w_net, ref.w_net, rtol=1e-7, atol=1e-8)
    assert np.isclose(stage.Q_evap, ref.Q_evap, rtol=1e-7, atol=1e-8)
    assert np.isclose(stage.Q_cond, ref.Q_cond, rtol=1e-7, atol=1e-8)
    assert np.isclose(stage.Q_heat, ref.Q_heat, rtol=1e-7, atol=1e-8)
    assert np.isclose(stage.Q_cool, ref.Q_cool, rtol=1e-7, atol=1e-8)


def test_cascade_requires_solution_before_dependent_properties():
    cycle = CascadeHeatPumpCycle()
    with pytest.raises(RuntimeError):
        _ = cycle.COP_h
    with pytest.raises(RuntimeError):
        cycle.build_stream_collection(include_cond=True)


def test_cascade_rejects_invalid_temperature_order():
    cycle = CascadeHeatPumpCycle()
    cycle.solve(
        T_evap=np.array([35.0, 25.0]),
        T_cond=np.array([30.0, 20.0]),
        refrigerant="R134a",
    )    
    assert cycle.solved == False


def test_cascade_num_cycles_matches_network_definition():
    cycle = CascadeHeatPumpCycle()
    T_cond = np.array([85.0, 60.0, 45.0])
    Q_heat = np.array([1000.0, 800.0, 200.0])
    T_evap = np.array([20.0, 0.0])
    Q_cool = np.array([600.0, None])   
    dt_cascade_hx = 4.0

    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dt_cascade_hx=dt_cascade_hx,
        dT_superheat=np.array([5.0, 5.0, 5.0, 5.0]),
        dT_subcool=np.array([2.0, 2.0, 2.0, 2.0]),
        dt_ihx_gas_side=np.array([5.0, 5.0, 5.0, 5.0]),
        refrigerant=["R134a", "R134a", "R134a", "R134a"],
        Q_heat=Q_heat,
        Q_cool=Q_cool,
    )

    assert cycle.num_cycles == len(T_cond) + len(T_evap) - 1
    assert len(cycle._subcycles) == cycle.num_cycles

    
    streams = StreamCollection()
    for subcycle in cycle.subcycles:
        streams = subcycle.build_stream_collection(include_cond=True, include_evap=True)
        assert np.isclose(sum([s.heat_flow for s in streams.get_cold_streams()]), subcycle.Q_cool, 0)
        assert np.isclose(sum([s.heat_flow for s in streams.get_hot_streams()]), subcycle.Q_heat, 0)

    streams = StreamCollection()
    streams = cycle.build_stream_collection(include_cond=True, include_evap=True)
    assert np.isclose(sum([s.heat_flow for s in streams.get_cold_streams()]), cycle.Q_cool, 0)
    assert np.isclose(sum([s.heat_flow for s in streams.get_hot_streams()]), cycle.Q_heat, 0)


def test_each_cascade_stage_matches_simple_heat_pump_solution():
    cycle = CascadeHeatPumpCycle()
    T_cond = np.array([80.0, 60.0])
    T_evap = np.array([20.0, 0.0])
    dt_cascade_hx = 5.0

    dT_superheat = np.array([6.0, 4.0, 2.0])
    dT_subcool = np.array([3.0, 2.0, 1.0])
    dt_ihx_gas_side = np.array([8.0, 6.0, 4.0])
    Q_heat = np.array([1200.0, 700.0])
    Q_cool = np.array([100.0, 900.0])
    Q_heat_all = np.array([1200.0, 700.0, 0.0])
    Q_cool_all = np.array([0.0, 100.0, 900.0])
    refrigerant = ["R134a", "R134a", "R134a"]
    eta_comp = 0.7

    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dt_cascade_hx=dt_cascade_hx,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dt_ihx_gas_side=dt_ihx_gas_side,
        refrigerant=refrigerant,
        eta_comp=eta_comp,
        Q_heat=Q_heat,
        Q_cool=Q_cool,
    )

    T_evap_all, T_cond_all = _cascade_stage_temperatures(T_evap, T_cond, dt_cascade_hx)

    for i in range(cycle.num_cycles):
        _assert_stage_matches_simple(
            cycle,
            i,
            T_evap=T_evap_all[i],
            T_cond=T_cond_all[i],
            dT_superheat=dT_superheat[i],
            dT_subcool=dT_subcool[i],
            dt_ihx_gas_side=dt_ihx_gas_side[i],
            refrigerant=refrigerant[i],
            eta_comp=eta_comp,
            Q_heat=Q_heat_all[i],
            Q_cool=Q_cool_all[i],
            Q_cas_heat=cycle.subcycles[i].Q_cas_heat,
        )


def test_cascade_aggregates_match_sum_of_stage_results():
    cycle = CascadeHeatPumpCycle()
    cycle.solve(
        T_evap=np.array([20.0, 0.0]),
        T_cond=np.array([80.0, 60.0]),
        dt_cascade_hx=5.0,
        dT_superheat=np.array([5.0, 5.0, 5.0]),
        dT_subcool=np.array([2.0, 2.0, 2.0]),
        dt_ihx_gas_side=np.array([5.0, 5.0, 5.0]),
        eta_comp=0.75,
        refrigerant=["R134a", "R134a", "R134a"],
        Q_heat=np.array([1200.0, 700.0, 0.0]),
        Q_cool=np.array([0.0, 0.0, 900.0]),
    )

    assert np.isclose(cycle.work, sum(c.work for c in cycle._subcycles), rtol=1e-7, atol=1e-8)
    assert np.isclose(cycle.Q_evap, sum(c.Q_evap for c in cycle._subcycles), rtol=1e-7, atol=1e-8)
    assert np.isclose(cycle.Q_cond, sum(c.Q_cond for c in cycle._subcycles), rtol=1e-7, atol=1e-8)
    assert np.isclose(cycle.Q_heat, sum(c.Q_heat for c in cycle._subcycles), rtol=1e-7, atol=1e-8)
    assert np.isclose(cycle.Q_cool, sum(c.Q_cool for c in cycle._subcycles), rtol=1e-7, atol=1e-8)


def test_cascade_build_stream_collection_is_union_of_stage_streams():
    cycle = CascadeHeatPumpCycle()
    cycle.solve(
        T_evap=np.array([20.0, 0.0]),
        T_cond=np.array([80.0, 60.0]),
        dt_cascade_hx=5.0,
        dT_superheat=np.array([5.0, 5.0, 5.0]),
        dT_subcool=np.array([2.0, 2.0, 2.0]),
        dt_ihx_gas_side=np.array([5.0, 5.0, 5.0]),
        eta_comp=0.75,
        refrigerant=["R134a", "R134a", "R134a"],
        Q_heat=np.array([1200.0, 700.0, 0.0]),
        Q_cool=np.array([0.0, 0.0, 900.0]),
    )

    combined = cycle.build_stream_collection(include_cond=True, include_evap=True)
    expected_count = sum(
        len(stage.build_stream_collection(include_cond=True, include_evap=True))
        for stage in cycle._subcycles
    )
    assert len(combined) == expected_count


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"refrigerant": ["R134a", "R134a"]},
        {"dt_ihx_gas_side": np.array([5.0, 5.0])},
    ],
)
def test_cascade_rejects_mismatched_per_stage_input_lengths(bad_kwargs):
    cycle = CascadeHeatPumpCycle()
    base = dict(
        T_evap=np.array([20.0, 0.0]),
        T_cond=np.array([80.0, 60.0]),
        dt_cascade_hx=5.0,
        dT_superheat=np.array([5.0, 5.0, 5.0]),
        dT_subcool=np.array([2.0, 2.0, 2.0]),
        Q_heat=np.array([1200.0, 700.0, 0.0]),
        Q_cool=np.array([0.0, 0.0, 900.0]),
        refrigerant=["R134a", "R134a", "R134a"],
        dt_ihx_gas_side=np.array([5.0, 5.0, 5.0]),
    )
    base.update(bad_kwargs)

    with pytest.raises(ValueError):
        cycle.solve(**base)


def test_cascade_solve_with_defaults_should_work_for_multistage():
    """Regression: multistage solve should work with documented defaults."""
    cycle = CascadeHeatPumpCycle()
    cycle.solve(
        T_evap=np.array([20.0, 0.0]),
        T_cond=np.array([80.0, 60.0]),
        dt_cascade_hx=5.0,
        refrigerant="R134a",
    )
    assert cycle.num_cycles == 3
    assert len(cycle.subcycles) == 3


def test_cascade_q_cool_last_nan_is_allowed_and_maps_to_q_evap():
    """Only the last stage may be unspecified for Q_cool."""
    cycle = CascadeHeatPumpCycle()
    cycle.solve(
        T_evap=np.array([20.0, 0.0]),
        T_cond=np.array([80.0, 60.0]),
        dt_cascade_hx=5.0,
        dT_superheat=np.array([5.0, 5.0, 5.0]),
        dT_subcool=np.array([2.0, 2.0, 2.0]),
        dt_ihx_gas_side=np.array([5.0, 5.0, 5.0]),
        eta_comp=0.75,
        refrigerant=["R134a", "R134a", "R134a"],
        Q_heat=np.array([1200.0, 700.0, 0.0]),
        Q_cool=np.array([0.0, 100.0, np.nan]),
    )

    last = cycle.subcycles[-1]
    assert np.isclose(last.Q_cool, last.Q_evap, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize(
    "bad_q_cool",
    [
        np.array([np.nan, 100.0, 300.0]),
        np.array([0.0, np.nan, 300.0]),
        np.array([None, 100.0, 300.0], dtype=object),
        np.array([0.0, None, 300.0], dtype=object),
    ],
)
def test_cascade_rejects_none_or_nan_q_cool_before_last_stage(bad_q_cool):
    cycle = CascadeHeatPumpCycle()
    with pytest.raises(ValueError):
        cycle.solve(
            T_evap=np.array([20.0, 0.0]),
            T_cond=np.array([80.0, 60.0]),
            dt_cascade_hx=5.0,
            dT_superheat=np.array([5.0, 5.0, 5.0]),
            dT_subcool=np.array([2.0, 2.0, 2.0]),
            dt_ihx_gas_side=np.array([5.0, 5.0, 5.0]),
            eta_comp=0.75,
            refrigerant=["R134a", "R134a", "R134a"],
            Q_heat=np.array([1200.0, 700.0, 0.0]),
            Q_cool=bad_q_cool,
        )


def test_cascade_q_heat_nan_defaults_to_one_for_first_and_zero_elsewhere():
    """Q_heat NaN defaults: index 0 -> 1.0, all other indices -> 0.0."""
    cycle = CascadeHeatPumpCycle()
    cycle.solve(
        T_evap=np.array([20.0, 0.0]),
        T_cond=np.array([80.0, 60.0]),
        dt_cascade_hx=5.0,
        dT_superheat=np.array([5.0, 5.0, 5.0]),
        dT_subcool=np.array([2.0, 2.0, 2.0]),
        dt_ihx_gas_side=np.array([5.0, 5.0, 5.0]),
        eta_comp=0.75,
        refrigerant=["R134a", "R134a", "R134a"],
        Q_heat=np.array([np.nan, np.nan]),
        Q_cool=np.array([0.0, 900.0]),
    )
    assert np.allclose(cycle.Q_heat_arr, np.array([1.0, 0.0, 0.0]), rtol=1e-7, atol=1e-8)
