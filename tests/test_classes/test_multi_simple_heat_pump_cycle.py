import numpy as np
import pytest

pytest.importorskip("CoolProp")

from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.classes.multi_simple_heat_pump import MultiSimpleHeatPumpCycle
from OpenPinch.classes.simple_heat_pump import SimpleHeatPumpCycle


def _assert_stage_matches_simple(parallel, i, *, T_evap, T_cond, **kwargs):
    ref = SimpleHeatPumpCycle()
    ref.solve(T_evap=T_evap, T_cond=T_cond, **kwargs)

    stage = parallel.subcycles[i]
    assert np.isclose(stage.w_net, ref.w_net, rtol=1e-7, atol=1e-8)
    assert np.isclose(stage.Q_evap, ref.Q_evap, rtol=1e-7, atol=1e-8)
    assert np.isclose(stage.Q_cond, ref.Q_cond, rtol=1e-7, atol=1e-8)
    assert np.isclose(stage.Q_heat, ref.Q_heat, rtol=1e-7, atol=1e-8)
    assert np.isclose(stage.Q_cool, ref.Q_cool, rtol=1e-7, atol=1e-8)


def test_parallel_requires_solution_before_dependent_properties():
    cycle = MultiSimpleHeatPumpCycle()
    with pytest.raises(RuntimeError):
        _ = cycle.COP_h
    with pytest.raises(RuntimeError):
        cycle.build_stream_collection(include_cond=True)


def test_parallel_rejects_mismatched_temperature_lengths():
    cycle = MultiSimpleHeatPumpCycle()
    with pytest.raises(ValueError):
        cycle.solve(
            T_evap=np.array([20.0, 0.0]),
            T_cond=np.array([80.0, 60.0, 45.0]),
            refrigerant="R134a",
        )


def test_parallel_solve_with_defaults_should_work_for_multiple_units():
    cycle = MultiSimpleHeatPumpCycle()
    cycle.solve(
        T_evap=np.array([20.0, 0.0]),
        T_cond=np.array([80.0, 60.0]),
        refrigerant="R134a",
    )
    assert cycle.num_cycles == 2
    assert len(cycle.subcycles) == 2


def test_each_parallel_stage_matches_simple_heat_pump_solution():
    cycle = MultiSimpleHeatPumpCycle()
    T_cond = np.array([80.0, 60.0])
    T_evap = np.array([20.0, 0.0])
    dT_superheat = np.array([6.0, 4.0])
    dT_subcool = np.array([3.0, 2.0])
    ihx_gas_dt = np.array([8.0, 6.0])
    Q_heat = np.array([1200.0, 700.0])
    Q_cool = np.array([900.0, 400.0])
    refrigerant = ["R134a", "R134a"]
    eta_comp = 0.7

    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        ihx_gas_dt=ihx_gas_dt,
        refrigerant=refrigerant,
        eta_comp=eta_comp,
        Q_heat=Q_heat,
        Q_cool=Q_cool,
    )

    for i in range(cycle.num_cycles):
        _assert_stage_matches_simple(
            cycle,
            i,
            T_evap=T_evap[i],
            T_cond=T_cond[i],
            dT_superheat=dT_superheat[i],
            dT_subcool=dT_subcool[i],
            ihx_gas_dt=ihx_gas_dt[i],
            refrigerant=refrigerant[i],
            eta_comp=eta_comp,
            Q_heat=Q_heat[i],
            Q_cool=Q_cool[i],
        )


def test_parallel_aggregates_match_sum_of_stage_results():
    cycle = MultiSimpleHeatPumpCycle()
    cycle.solve(
        T_evap=np.array([20.0, 0.0]),
        T_cond=np.array([80.0, 60.0]),
        dT_superheat=np.array([5.0, 5.0]),
        dT_subcool=np.array([2.0, 2.0]),
        ihx_gas_dt=np.array([5.0, 5.0]),
        eta_comp=0.75,
        refrigerant=["R134a", "R134a"],
        Q_heat=np.array([1200.0, 700.0]),
        Q_cool=np.array([900.0, 500.0]),
    )

    assert np.isclose(cycle.work, sum(c.work for c in cycle.subcycles), rtol=1e-7, atol=1e-8)
    assert np.isclose(cycle.Q_evap, sum(c.Q_evap for c in cycle.subcycles), rtol=1e-7, atol=1e-8)
    assert np.isclose(cycle.Q_cond, sum(c.Q_cond for c in cycle.subcycles), rtol=1e-7, atol=1e-8)
    assert np.isclose(cycle.Q_heat, sum(c.Q_heat for c in cycle.subcycles), rtol=1e-7, atol=1e-8)
    assert np.isclose(cycle.Q_cool, sum(c.Q_cool for c in cycle.subcycles), rtol=1e-7, atol=1e-8)


def test_parallel_build_stream_collection_is_union_of_stage_streams():
    cycle = MultiSimpleHeatPumpCycle()
    cycle.solve(
        T_evap=np.array([20.0, 0.0]),
        T_cond=np.array([80.0, 60.0]),
        dT_superheat=np.array([5.0, 5.0]),
        dT_subcool=np.array([2.0, 2.0]),
        ihx_gas_dt=np.array([5.0, 5.0]),
        eta_comp=0.75,
        refrigerant=["R134a", "R134a"],
        Q_heat=np.array([1200.0, 700.0]),
        Q_cool=np.array([900.0, 500.0]),
    )

    combined = cycle.build_stream_collection(include_cond=True, include_evap=True)
    expected_count = sum(
        len(stage.build_stream_collection(include_cond=True, include_evap=True))
        for stage in cycle.subcycles
    )
    assert len(combined) == expected_count


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"refrigerant": ["R134a", "R134a", "R134a"]},
        {"ihx_gas_dt": np.array([5.0, 5.0, 5.0])},
        {"dT_superheat": np.array([5.0, 5.0, 5.0])},
        {"dT_subcool": np.array([2.0, 2.0, 2.0])},
    ],
)
def test_parallel_rejects_mismatched_per_stage_input_lengths(bad_kwargs):
    cycle = MultiSimpleHeatPumpCycle()
    base = dict(
        T_evap=np.array([20.0, 0.0]),
        T_cond=np.array([80.0, 60.0]),
        Q_heat=np.array([1200.0, 700.0]),
        Q_cool=np.array([900.0, 500.0]),
        refrigerant=["R134a", "R134a"],
        ihx_gas_dt=np.array([5.0, 5.0]),
        dT_superheat=np.array([5.0, 5.0]),
        dT_subcool=np.array([2.0, 2.0]),
    )
    base.update(bad_kwargs)

    with pytest.raises(ValueError):
        cycle.solve(**base)


def test_parallel_q_cool_nan_and_none_are_allowed_for_any_cycle():
    cycle = MultiSimpleHeatPumpCycle()
    cycle.solve(
        T_evap=np.array([20.0, 0.0]),
        T_cond=np.array([80.0, 60.0]),
        dT_superheat=np.array([5.0, 5.0]),
        dT_subcool=np.array([2.0, 2.0]),
        ihx_gas_dt=np.array([5.0, 5.0]),
        eta_comp=0.75,
        refrigerant=["R134a", "R134a"],
        Q_heat=np.array([1200.0, 700.0]),
        Q_cool=np.array([np.nan, None], dtype=object),
    )

    for stage in cycle.subcycles:
        assert np.isclose(stage.Q_cool, stage.Q_evap, rtol=1e-7, atol=1e-8)


def test_parallel_q_heat_nan_defaults_to_one_for_each_stage():
    cycle = MultiSimpleHeatPumpCycle()
    cycle.solve(
        T_evap=np.array([20.0, 0.0]),
        T_cond=np.array([80.0, 60.0]),
        dT_superheat=np.array([5.0, 5.0]),
        dT_subcool=np.array([2.0, 2.0]),
        ihx_gas_dt=np.array([5.0, 5.0]),
        eta_comp=0.75,
        refrigerant=["R134a", "R134a"],
        Q_heat=np.array([np.nan, np.nan]),
        Q_cool=np.array([900.0, 500.0]),
    )
    assert np.allclose(cycle.Q_heat_arr, np.array([1.0, 1.0]), rtol=1e-7, atol=1e-8)


def test_parallel_streams_match_aggregate_heat():
    cycle = MultiSimpleHeatPumpCycle()
    cycle.solve(
        T_evap=np.array([20.0, 0.0]),
        T_cond=np.array([80.0, 60.0]),
        dT_superheat=np.array([5.0, 5.0]),
        dT_subcool=np.array([2.0, 2.0]),
        ihx_gas_dt=np.array([5.0, 5.0]),
        eta_comp=0.75,
        refrigerant=["R134a", "R134a"],
        Q_heat=np.array([1200.0, 700.0]),
        Q_cool=np.array([900.0, 500.0]),
    )

    streams = StreamCollection()
    streams = cycle.build_stream_collection(include_cond=True, include_evap=True)
    assert np.isclose(sum([s.heat_flow for s in streams.get_cold_streams()]), cycle.Q_cool, 0)
    assert np.isclose(sum([s.heat_flow for s in streams.get_hot_streams()]), cycle.Q_heat, 0)
