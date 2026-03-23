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
    with pytest.raises(ValueError):
        cycle.solve(
            T_evap=np.array([35.0, 25.0]),
            T_cond=np.array([30.0, 20.0]),
            refrigerant="R134a",
        )


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
        ihx_gas_dt=np.array([5.0, 5.0, 5.0, 5.0]),
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
    ihx_gas_dt = np.array([8.0, 6.0, 4.0])
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
        ihx_gas_dt=ihx_gas_dt,
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
            ihx_gas_dt=ihx_gas_dt[i],
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
        ihx_gas_dt=np.array([5.0, 5.0, 5.0]),
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
        ihx_gas_dt=np.array([5.0, 5.0, 5.0]),
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
        {"ihx_gas_dt": np.array([5.0, 5.0])},
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
        ihx_gas_dt=np.array([5.0, 5.0, 5.0]),
    )
    base.update(bad_kwargs)

    with pytest.raises(ValueError):
        cycle.solve(**base)
