"""Regression tests for the multi simple heat pump cycle classes."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("CoolProp")

import OpenPinch.classes.multi_simple_heat_pump as multi_mod
from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.classes.multi_simple_heat_pump import MultiSimpleHeatPumpCycle
from OpenPinch.classes.simple_heat_pump import SimpleHeatPumpCycle


def _assert_stage_matches_simple(parallel, i, *, T_evap, T_cond, **kwargs):
    """Assert that stage matches simple for this test module."""
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
    dt_ihx_gas_side = np.array([8.0, 6.0])
    Q_heat = np.array([1200.0, 700.0])
    Q_cool = np.array([900.0, 400.0])
    refrigerant = ["R134a", "R134a"]
    eta_comp = 0.7

    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dt_ihx_gas_side=dt_ihx_gas_side,
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
            dt_ihx_gas_side=dt_ihx_gas_side[i],
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
        dt_ihx_gas_side=np.array([5.0, 5.0]),
        eta_comp=0.75,
        refrigerant=["R134a", "R134a"],
        Q_heat=np.array([1200.0, 700.0]),
        Q_cool=np.array([900.0, 500.0]),
    )

    assert np.isclose(
        cycle.work, sum(c.work for c in cycle.subcycles), rtol=1e-7, atol=1e-8
    )
    assert np.isclose(
        cycle.Q_evap, sum(c.Q_evap for c in cycle.subcycles), rtol=1e-7, atol=1e-8
    )
    assert np.isclose(
        cycle.Q_cond, sum(c.Q_cond for c in cycle.subcycles), rtol=1e-7, atol=1e-8
    )
    assert np.isclose(
        cycle.Q_heat, sum(c.Q_heat for c in cycle.subcycles), rtol=1e-7, atol=1e-8
    )
    assert np.isclose(
        cycle.Q_cool, sum(c.Q_cool for c in cycle.subcycles), rtol=1e-7, atol=1e-8
    )


def test_parallel_build_stream_collection_is_union_of_stage_streams():
    cycle = MultiSimpleHeatPumpCycle()
    cycle.solve(
        T_evap=np.array([20.0, 0.0]),
        T_cond=np.array([80.0, 60.0]),
        dT_superheat=np.array([5.0, 5.0]),
        dT_subcool=np.array([2.0, 2.0]),
        dt_ihx_gas_side=np.array([5.0, 5.0]),
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
        {"dt_ihx_gas_side": np.array([5.0, 5.0, 5.0])},
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
        dt_ihx_gas_side=np.array([5.0, 5.0]),
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
        dt_ihx_gas_side=np.array([5.0, 5.0]),
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
        dt_ihx_gas_side=np.array([5.0, 5.0]),
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
        dt_ihx_gas_side=np.array([5.0, 5.0]),
        eta_comp=0.75,
        refrigerant=["R134a", "R134a"],
        Q_heat=np.array([1200.0, 700.0]),
        Q_cool=np.array([900.0, 500.0]),
    )

    streams = StreamCollection()
    streams = cycle.build_stream_collection(include_cond=True, include_evap=True)
    assert np.isclose(
        sum([s.heat_flow for s in streams.get_cold_streams()]), cycle.Q_cool, 0
    )
    assert np.isclose(
        sum([s.heat_flow for s in streams.get_hot_streams()]), cycle.Q_heat, 0
    )


def _fake_network_cycle(
    *,
    solved=True,
    refrigerant="R134A",
    T_evap=20.0,
    T_cond=80.0,
    dT_superheat=5.0,
    dT_subcool=2.0,
    eta_comp=0.75,
    dt_ihx_gas_side=5.0,
    Q_evap=60.0,
    Q_cond=100.0,
    Q_cool=55.0,
    Q_heat=95.0,
    Q_cas_cool=5.0,
    Q_cas_heat=6.0,
    work=40.0,
    penalty=1.0,
):
    return SimpleNamespace(
        solved=solved,
        refrigerant=refrigerant,
        T_evap=T_evap,
        T_cond=T_cond,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        eta_comp=eta_comp,
        dt_ihx_gas_side=dt_ihx_gas_side,
        Q_evap=Q_evap,
        Q_cond=Q_cond,
        Q_cool=Q_cool,
        Q_heat=Q_heat,
        Q_cas_cool=Q_cas_cool,
        Q_cas_heat=Q_cas_heat,
        work=work,
        penalty=penalty,
    )


class _DummyMultiCycle:
    solved = True
    refrigerant = "water"
    T_evap = 20.0
    T_cond = 80.0
    dT_superheat = 5.0
    dT_subcool = 3.0
    eta_comp = 0.7
    dt_ihx_gas_side = 10.0
    Q_evap = 10.0
    Q_cas_cool = 2.0
    Q_cool = 3.0
    Q_cond = 12.0
    Q_cas_heat = 4.0
    Q_heat = 8.0
    work = 5.0
    penalty = 0.5

    def solve(self, **kwargs):
        self.Q_heat = kwargs.get("Q_heat", self.Q_heat)
        self.Q_cas_cool = kwargs.get("Q_cool", self.Q_cas_cool)
        return self.work


def test_multi_simple_property_sweep_and_normalization_branches():
    cycle = MultiSimpleHeatPumpCycle()
    c1 = _fake_network_cycle()
    c2 = _fake_network_cycle(
        T_evap=15.0, T_cond=75.0, work=30.0, Q_evap=50.0, Q_cond=90.0
    )
    cycle._subcycles = [c1, c2]
    cycle._solved = True

    assert cycle.Q_evap == 110.0
    assert cycle.Q_cas_cool == 10.0
    assert cycle.Q_cool == 110.0
    assert cycle.Q_cond == 190.0
    assert cycle.Q_cas_heat == 12.0
    assert cycle.Q_heat == 190.0
    assert cycle.work == 70.0
    assert cycle.penalty == 2.0
    assert cycle.COP_h == pytest.approx(cycle.Q_heat / cycle.work)
    assert cycle.COP_r == pytest.approx(cycle.Q_cool / cycle.work)
    assert cycle.COP_o == pytest.approx((cycle.Q_heat + cycle.Q_cool) / cycle.work)
    assert cycle.refrigerant.tolist() == ["R134A", "R134A"]
    assert cycle.T_evap.tolist() == [20.0, 15.0]
    assert cycle.T_cond.tolist() == [80.0, 75.0]
    assert cycle.dT_superheat.tolist() == [5.0, 5.0]
    assert cycle.dT_subcool.tolist() == [2.0, 2.0]
    assert cycle.eta_comp.tolist() == [0.75, 0.75]
    assert cycle.dt_ihx_gas_side.tolist() == [5.0, 5.0]
    assert cycle.num_cycles == 1
    assert len(cycle.subcycles) == 2

    cycle._subcycles = [_fake_network_cycle(work=0.0), _fake_network_cycle(work=0.0)]
    with pytest.raises(ZeroDivisionError):
        _ = cycle.COP_h
    with pytest.raises(ZeroDivisionError):
        _ = cycle.COP_r
    with pytest.raises(ZeroDivisionError):
        _ = cycle.COP_o

    assert cycle._as_1d_numeric_array(None).shape == (1,)
    assert cycle._as_1d_numeric_array(np.nan).shape == (1,)
    with pytest.raises(ValueError):
        cycle._as_1d_numeric_array(np.array([[1.0]]))

    t_e, t_c = cycle._normalize_temperature_arrays(
        np.array([20.0]), np.array([80.0, 70.0])
    )
    assert t_e.shape == t_c.shape == (2,)
    t_e2, t_c2 = cycle._normalize_temperature_arrays(
        np.array([20.0, 10.0]), np.array([80.0])
    )
    assert t_e2.shape == t_c2.shape == (2,)
    with pytest.raises(ValueError):
        cycle._normalize_temperature_arrays(np.array([np.nan]), np.array([80.0]))
    with pytest.raises(ValueError):
        cycle._normalize_temperature_arrays(np.array([30.0]), np.array([20.0]))
    with pytest.raises(ValueError):
        cycle._normalize_temperature_arrays(
            np.array([30.0, 20.0]), np.array([70.0, 60.0, 50.0])
        )

    assert cycle._normalize_per_cycle_array(np.array([1.0]), 2, name="x").shape == (2,)
    assert cycle._normalize_per_cycle_array(
        np.array([1.0, 2.0]), 2, name="x"
    ).shape == (2,)
    with pytest.raises(ValueError, match="x must be scalar"):
        cycle._normalize_per_cycle_array(np.array([1.0, 2.0, 3.0]), 2, name="x")

    assert cycle._normalize_Q_heat(None, 2).tolist() == [1.0, 1.0]
    assert cycle._normalize_Q_heat(np.array([np.nan, 2.0]), 2).tolist() == [1.0, 2.0]
    with pytest.raises(ValueError):
        cycle._normalize_Q_heat(np.array([1.0, 2.0, 3.0]), 2)

    assert cycle._normalize_Q_cool(None, 2).tolist() == [None, None]
    assert cycle._normalize_Q_cool(np.array([5.0]), 2).tolist() == [5.0, 5.0]
    assert cycle._normalize_Q_cool(
        np.array([np.nan, None], dtype=object), 2
    ).tolist() == [None, None]
    with pytest.raises(ValueError):
        cycle._normalize_Q_cool(np.array([1.0, 2.0, 3.0]), 2)
    with pytest.raises(ValueError):
        cycle._normalize_Q_cool(np.array([object(), 1.0], dtype=object), 2)


def test_multi_simple_property_arrays_and_helper_error_branches():
    hp = MultiSimpleHeatPumpCycle()
    hp._subcycles = [_DummyMultiCycle(), _DummyMultiCycle()]
    hp._solved = True

    assert hp.Q_evap_arr.shape == (2,)
    assert hp.Q_cas_cool_arr.shape == (2,)
    assert hp.Q_cool_arr.shape == (2,)
    assert hp.Q_cond_arr.shape == (2,)
    assert hp.Q_cas_heat_arr.shape == (2,)
    assert hp.Q_heat_arr.shape == (2,)
    assert hp.work_arr.shape == (2,)
    assert hp.dtcont == 0.0
    assert hp.dt_diff_max == 0.5
    assert hp.penalty > 0

    hp._solved = False
    assert hp.penalty == 0

    with pytest.raises(ValueError, match="Input must be numeric"):
        hp._as_1d_numeric_array("bad")

    with pytest.raises(ValueError, match="must be scalar or have matching lengths"):
        hp._normalize_temperature_arrays([10.0, 20.0], [30.0, 40.0, 50.0])

    with pytest.raises(
        ValueError, match="must be scalar or have one value per heat pump"
    ):
        hp._normalize_per_cycle_array([1.0, 2.0], n_cycles=3, name="test")

    with pytest.raises(ValueError, match="Q_cool values must be numeric"):
        hp._normalize_Q_cool([object(), None], n_cycles=2)

    assert hp._normalize_Q_cool(np.nan, n_cycles=2).tolist() == [None, None]
    assert hp._normalize_Q_cool(5.0, n_cycles=2).tolist() == [5.0, 5.0]
    with pytest.raises(ValueError, match="Incompatible Q_cool input"):
        hp._normalize_Q_cool(np.array([[1.0, 2.0]]), n_cycles=2)


def test_multi_simple_solve_single_refrigerant_list_branch(monkeypatch):
    hp = MultiSimpleHeatPumpCycle()
    monkeypatch.setattr(multi_mod, "SimpleHeatPumpCycle", _DummyMultiCycle)

    work = hp.solve(
        T_evap=np.array([20.0, 15.0]),
        T_cond=np.array([60.0, 55.0]),
        refrigerant=["water"],
        Q_heat=np.array([1.0, 2.0]),
        Q_cool=np.array([None, None]),
    )

    assert isinstance(work, float)
    assert hp.solved is True
