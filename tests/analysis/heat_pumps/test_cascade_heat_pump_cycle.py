"""Regression tests for the cascade heat pump cycle classes."""

from types import SimpleNamespace

import numpy as np
import pytest

import OpenPinch.analysis.heat_pumps.cycles.cascade_vapour_compression_cycle as cascade_mod
from OpenPinch.analysis.heat_pumps.common.encoding import (
    DutyAllocationRequest,
    StageDutyRequest,
)
from OpenPinch.analysis.heat_pumps.cycles.cascade_vapour_compression_cycle import (
    CascadeVapourCompressionCycle,
)
from OpenPinch.analysis.heat_pumps.cycles.vapour_compression_cycle import (
    VapourCompressionCycle,
)
from OpenPinch.domain.stream_collection import StreamCollection

pytest.importorskip("CoolProp")


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
    ref = VapourCompressionCycle()
    kwargs.setdefault("dtcont", 0.0)
    ref.solve(T_evap=T_evap, T_cond=T_cond, **kwargs)

    stage = cascade._subcycles[i]
    assert np.isclose(stage.w_net, ref.w_net, rtol=1e-7, atol=1e-8)
    assert np.isclose(stage.Q_evap, ref.Q_evap, rtol=1e-7, atol=1e-8)
    assert np.isclose(stage.Q_cond, ref.Q_cond, rtol=1e-7, atol=1e-8)
    assert np.isclose(stage.Q_heat, ref.Q_heat, rtol=1e-7, atol=1e-8)


def test_cascade_requires_solution_before_dependent_properties():
    cycle = CascadeVapourCompressionCycle()
    with pytest.raises(RuntimeError):
        _ = cycle.COP_h
    with pytest.raises(RuntimeError):
        cycle.build_stream_collection(include_cond=True)


def test_cascade_rejects_invalid_temperature_order():
    cycle = CascadeVapourCompressionCycle()
    cycle.solve(
        T_evap=np.array([35.0, 25.0]),
        T_cond=np.array([30.0, 20.0]),
        dtcont=0.0,
        refrigerant="R134a",
    )
    assert not cycle.solved


def test_cascade_returns_penalty_when_aggregate_work_is_invalid(monkeypatch):
    class NegativeWorkCycle:
        solved = True
        work = -4.0
        Q_cas_cool = 0.0
        Q_cas_heat = 0.0

        def solve(self, **_kwargs):
            return None

    monkeypatch.setattr(cascade_mod, "VapourCompressionCycle", NegativeWorkCycle)
    cycle = CascadeVapourCompressionCycle()

    work = cycle.solve(
        T_evap=np.array([0.0]),
        T_cond=np.array([60.0]),
        dtcont=0.0,
        refrigerant="R134a",
    )

    assert work == pytest.approx(4.0)
    assert not cycle.solved


def test_cascade_num_cycles_matches_network_definition():
    cycle = CascadeVapourCompressionCycle()
    T_cond = np.array([85.0, 60.0, 45.0])
    Q_heat = np.array([1000.0, 800.0, 200.0])
    T_evap = np.array([20.0, 0.0])
    Q_cool = np.array([600.0, None])
    dt_cascade_hx = 4.0

    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dtcont=0.0,
        dt_cascade_hx=dt_cascade_hx,
        dT_superheat=np.array([5.0, 5.0, 5.0, 5.0]),
        dT_subcool=np.array([2.0, 2.0, 2.0, 2.0]),
        dT_ihx_gas_side=np.array([5.0, 5.0, 5.0, 5.0]),
        refrigerant=["R134a", "R134a", "R134a", "R134a"],
        Q_heat=Q_heat,
        Q_cool=Q_cool,
        is_heat_pump=True,
    )

    assert cycle.num_cycles == len(T_cond) + len(T_evap) - 1
    assert len(cycle._subcycles) == cycle.num_cycles

    streams = StreamCollection()
    for subcycle in cycle.subcycles:
        streams = subcycle.build_stream_collection(include_cond=True, include_evap=True)
        assert np.isclose(
            sum([s.heat_flow for s in streams.get_cold_streams()]),
            subcycle.Q_cool,
            rtol=1e-7,
            atol=1e-8,
        )
        assert np.isclose(
            sum([s.heat_flow for s in streams.get_hot_streams()]),
            subcycle.Q_heat,
            rtol=1e-7,
            atol=1e-8,
        )

    streams = StreamCollection()
    streams = cycle.build_stream_collection(include_cond=True, include_evap=True)
    assert np.isclose(
        sum([s.heat_flow for s in streams.get_cold_streams()]),
        cycle.Q_cool,
        rtol=1e-7,
        atol=1e-8,
    )
    assert np.isclose(
        sum([s.heat_flow for s in streams.get_hot_streams()]),
        cycle.Q_heat,
        rtol=1e-7,
        atol=1e-8,
    )


def test_each_cascade_stage_matches_simple_heat_pump_solution():
    cycle = CascadeVapourCompressionCycle()
    T_cond = np.array([80.0, 60.0])
    T_evap = np.array([20.0, 0.0])
    dt_cascade_hx = 5.0

    dT_superheat = np.array([6.0, 4.0, 2.0])
    dT_subcool = np.array([3.0, 2.0, 1.0])
    dT_ihx_gas_side = np.array([8.0, 6.0, 4.0])
    Q_heat = np.array([1200.0, 700.0])
    Q_cool = np.array([100.0, 900.0])
    Q_heat_all = np.array([1200.0, 700.0, 0.0])
    Q_cool_all = np.array([0.0, 100.0, 900.0])
    refrigerant = ["R134a", "R134a", "R134a"]
    eta_comp = 0.7

    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dtcont=0.0,
        dt_cascade_hx=dt_cascade_hx,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dT_ihx_gas_side=dT_ihx_gas_side,
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
            dT_ihx_gas_side=dT_ihx_gas_side[i],
            refrigerant=refrigerant[i],
            eta_comp=eta_comp,
            Q_heat=Q_heat_all[i],
            Q_cool=Q_cool_all[i],
            Q_cas_heat=cycle.subcycles[i].Q_cas_heat,
        )


def test_cascade_aggregates_match_sum_of_stage_results():
    cycle = CascadeVapourCompressionCycle()
    cycle.solve(
        T_evap=np.array([20.0, 0.0]),
        T_cond=np.array([80.0, 60.0]),
        dtcont=0.0,
        dt_cascade_hx=5.0,
        dT_superheat=np.array([5.0, 5.0, 5.0]),
        dT_subcool=np.array([2.0, 2.0, 2.0]),
        dT_ihx_gas_side=np.array([5.0, 5.0, 5.0]),
        eta_comp=0.75,
        refrigerant=["R134a", "R134a", "R134a"],
        Q_heat=np.array([1200.0, 700.0, 0.0]),
        Q_cool=np.array([0.0, 0.0, 900.0]),
    )

    assert np.isclose(
        cycle.work, sum(c.work for c in cycle._subcycles), rtol=1e-7, atol=1e-8
    )
    assert np.isclose(
        cycle.Q_evap, sum(c.Q_evap for c in cycle._subcycles), rtol=1e-7, atol=1e-8
    )
    assert np.isclose(
        cycle.Q_cond, sum(c.Q_cond for c in cycle._subcycles), rtol=1e-7, atol=1e-8
    )
    assert np.isclose(
        cycle.Q_heat, sum(c.Q_heat for c in cycle._subcycles), rtol=1e-7, atol=1e-8
    )
    assert np.isclose(
        cycle.Q_cool, sum(c.Q_cool for c in cycle._subcycles), rtol=1e-7, atol=1e-8
    )


def test_cascade_build_stream_collection_is_union_of_stage_streams():
    cycle = CascadeVapourCompressionCycle()
    cycle.solve(
        T_evap=np.array([20.0, 0.0]),
        T_cond=np.array([80.0, 60.0]),
        dtcont=0.0,
        dt_cascade_hx=5.0,
        dT_superheat=np.array([5.0, 5.0, 5.0]),
        dT_subcool=np.array([2.0, 2.0, 2.0]),
        dT_ihx_gas_side=np.array([5.0, 5.0, 5.0]),
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
        {"dT_ihx_gas_side": np.array([5.0, 5.0])},
    ],
)
def test_cascade_rejects_mismatched_per_stage_input_lengths(bad_kwargs):
    cycle = CascadeVapourCompressionCycle()
    base = dict(
        T_evap=np.array([20.0, 0.0]),
        T_cond=np.array([80.0, 60.0]),
        dtcont=0.0,
        dt_cascade_hx=5.0,
        dT_superheat=np.array([5.0, 5.0, 5.0]),
        dT_subcool=np.array([2.0, 2.0, 2.0]),
        Q_heat=np.array([1200.0, 700.0, 0.0]),
        Q_cool=np.array([0.0, 0.0, 900.0]),
        refrigerant=["R134a", "R134a", "R134a"],
        dT_ihx_gas_side=np.array([5.0, 5.0, 5.0]),
    )
    base.update(bad_kwargs)

    with pytest.raises(ValueError):
        cycle.solve(**base)


def test_cascade_solve_with_defaults_gives_positive_work_in_every_stage():
    # The default 10 K IHX once compressed from a mismatched (h, s) pair and
    # gave the top stage negative work; every stage must now need power.
    cycle = CascadeVapourCompressionCycle()
    work = cycle.solve(
        T_evap=np.array([20.0, 0.0]),
        T_cond=np.array([80.0, 60.0]),
        dtcont=0.0,
        dt_cascade_hx=5.0,
        refrigerant="R134a",
    )
    assert cycle.num_cycles == 3
    assert cycle.solved is True
    assert np.all(cycle.work_arr > 0.0)
    assert work == pytest.approx(cycle.work_arr.sum())


def test_cascade_q_cool_last_nan_is_allowed_and_maps_to_q_evap():
    """Only the last stage may be unspecified for Q_cool."""
    cycle = CascadeVapourCompressionCycle()
    cycle.solve(
        T_evap=np.array([20.0, 0.0]),
        T_cond=np.array([80.0, 60.0]),
        dtcont=0.0,
        dt_cascade_hx=5.0,
        dT_superheat=np.array([5.0, 5.0, 5.0]),
        dT_subcool=np.array([2.0, 2.0, 2.0]),
        dT_ihx_gas_side=np.array([5.0, 5.0, 5.0]),
        eta_comp=0.75,
        refrigerant=["R134a", "R134a", "R134a"],
        Q_heat=np.array([1200.0, 700.0, 0.0]),
        Q_cool=np.array([0.0, 100.0, np.nan]),
    )

    last = cycle.subcycles[-1]
    assert np.isclose(last.Q_cool, last.Q_evap, rtol=1e-7, atol=1e-8)


def test_each_cascade_stage_matches_simple_refrigeration_solution():
    cycle = CascadeVapourCompressionCycle()
    T_cond = np.array([80.0, 60.0])
    T_evap = np.array([20.0, 0.0])
    dt_cascade_hx = 5.0
    dT_superheat = np.array([6.0, 4.0, 2.0])
    dT_subcool = np.array([3.0, 2.0, 1.0])
    dT_ihx_gas_side = np.array([8.0, 6.0, 4.0])
    Q_heat = np.array([500.0, 1e9, 50.0])
    Q_cool = np.array([100.0, 900.0])
    Q_heat_all = np.array([500.0, 1e9, 50.0])
    Q_cool_all = np.array([0.0, 100.0, 900.0])
    refrigerant = ["R134a", "R134a", "R134a"]
    eta_comp = 0.7

    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dtcont=0.0,
        dt_cascade_hx=dt_cascade_hx,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dT_ihx_gas_side=dT_ihx_gas_side,
        refrigerant=refrigerant,
        eta_comp=eta_comp,
        Q_heat=Q_heat,
        Q_cool=Q_cool,
        is_heat_pump=False,
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
            dT_ihx_gas_side=dT_ihx_gas_side[i],
            refrigerant=refrigerant[i],
            eta_comp=eta_comp,
            Q_heat=Q_heat_all[i],
            Q_cool=Q_cool_all[i],
            Q_cas_cool=cycle.subcycles[i].Q_cas_cool,
            is_heat_pump=False,
        )

    for stage in cycle.subcycles:
        assert np.isclose(stage.Q_evap, stage.Q_cool + stage.Q_cas_cool, 0.0)
        assert stage.Q_heat <= stage.Q_cond + 1e-6
        assert stage.Q_cas_heat >= -1e-8
        assert np.isclose(
            stage.Q_cond,
            stage.Q_heat + stage.Q_cas_heat,
            rtol=1e-7,
            atol=1e-8,
        )

    assert np.isclose(cycle.subcycles[0].Q_heat, 0.0, 0.0)
    assert np.isclose(cycle.subcycles[1].Q_heat, cycle.subcycles[1].Q_cond, 0.0)


def test_cascade_refrigeration_streams_and_aggregates_match():
    cycle = CascadeVapourCompressionCycle()
    cycle.solve(
        T_evap=np.array([20.0, 0.0]),
        T_cond=np.array([80.0, 60.0]),
        dtcont=0.0,
        dt_cascade_hx=5.0,
        dT_superheat=np.array([6.0, 4.0, 2.0]),
        dT_subcool=np.array([3.0, 2.0, 1.0]),
        dT_ihx_gas_side=np.array([8.0, 6.0, 4.0]),
        eta_comp=0.7,
        refrigerant=["R134a", "R134a", "R134a"],
        Q_heat=np.array([500.0, 1e9, 50.0]),
        Q_cool=np.array([100.0, 900.0]),
        is_heat_pump=False,
    )

    assert np.isclose(
        cycle.work, sum(c.work for c in cycle._subcycles), rtol=1e-7, atol=1e-8
    )
    assert np.isclose(
        cycle.Q_evap, sum(c.Q_evap for c in cycle._subcycles), rtol=1e-7, atol=1e-8
    )
    assert np.isclose(
        cycle.Q_cond, sum(c.Q_cond for c in cycle._subcycles), rtol=1e-7, atol=1e-8
    )
    assert np.isclose(
        cycle.Q_heat, sum(c.Q_heat for c in cycle._subcycles), rtol=1e-7, atol=1e-8
    )
    assert np.isclose(
        cycle.Q_cool, sum(c.Q_cool for c in cycle._subcycles), rtol=1e-7, atol=1e-8
    )
    assert cycle.COP_r > 0.0

    streams = StreamCollection()
    streams = cycle.build_stream_collection(include_cond=True, include_evap=True)
    assert np.isclose(
        sum([s.heat_flow for s in streams.get_cold_streams()]),
        cycle.Q_cool,
        rtol=1e-7,
        atol=1e-8,
    )
    assert np.isclose(
        sum([s.heat_flow for s in streams.get_hot_streams()]),
        cycle.Q_heat,
        rtol=1e-7,
        atol=1e-8,
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
    dT_ihx_gas_side=5.0,
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
        dT_ihx_gas_side=dT_ihx_gas_side,
        Q_evap=Q_evap,
        Q_cond=Q_cond,
        Q_cool=Q_cool,
        Q_heat=Q_heat,
        Q_cas_cool=Q_cas_cool,
        Q_cas_heat=Q_cas_heat,
        work=work,
        penalty=penalty,
    )


class _DummyCascadeCycle:
    solved = True
    refrigerant = "water"
    T_evap = 20.0
    T_cond = 80.0
    dT_superheat = 5.0
    dT_subcool = 3.0
    eta_comp = 0.7
    dT_ihx_gas_side = 10.0
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


def test_cascade_property_sweep_and_normalization_branches():
    cycle = CascadeVapourCompressionCycle()
    c1 = _fake_network_cycle()
    c2 = _fake_network_cycle(
        T_evap=10.0, T_cond=70.0, work=20.0, Q_evap=30.0, Q_cond=50.0
    )
    cycle._subcycles = [c1, c2]
    cycle._solved = True
    cycle._dt_cascade_hx = 3.0

    assert cycle.Q_evap == 90.0
    assert cycle.Q_cas_cool == 10.0
    assert cycle.Q_cool == 110.0
    assert cycle.Q_cond == 150.0
    assert cycle.Q_cas_heat == 12.0
    assert cycle.Q_heat == 190.0
    assert cycle.work == 60.0
    assert cycle.penalty == 2.0
    assert cycle.COP_h == pytest.approx(cycle.Q_heat / cycle.work)
    assert cycle.COP_r == pytest.approx(cycle.Q_cool / cycle.work)
    assert cycle.COP_o == pytest.approx((cycle.Q_heat + cycle.Q_cool) / cycle.work)
    assert cycle.dt_cascade_hx == 3.0
    assert cycle.refrigerant.tolist() == ["R134A", "R134A"]
    assert cycle.T_evap.tolist() == [20.0, 10.0]
    assert cycle.T_cond.tolist() == [80.0, 70.0]
    assert cycle.dT_superheat.tolist() == [5.0, 5.0]
    assert cycle.dT_subcool.tolist() == [2.0, 2.0]
    assert cycle.eta_comp.tolist() == [0.75, 0.75]
    assert cycle.dT_ihx_gas_side.tolist() == [5.0, 5.0]
    assert cycle.num_cycles == 1
    assert len(cycle.subcycles) == 2

    cycle._solved = False
    cycle._max_work = 123.0
    assert cycle.work == 123.0
    assert cycle.penalty == 0

    cycle._solved = True
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
        cycle._as_1d_numeric_array(np.array([[1.0, 2.0]]))

    assert cycle._normalize_dT_superheat(np.array([1.0]), 2, 2).shape == (3,)
    assert cycle._normalize_dT_superheat(np.array([1.0, 2.0]), 2, 2).shape == (3,)
    with pytest.raises(ValueError):
        cycle._normalize_dT_superheat(np.array([1.0, 2.0, 3.0, 4.0]), 2, 2)

    assert cycle._normalize_dT_subcool(np.array([1.0]), 2, 2).shape == (3,)
    assert cycle._normalize_dT_subcool(np.array([1.0, 2.0]), 2, 2).shape == (3,)
    with pytest.raises(ValueError):
        cycle._normalize_dT_subcool(np.array([1.0, 2.0, 3.0, 4.0]), 2, 2)

    qh = cycle._normalize_Q_heat(np.array([1.0, np.nan]), 2, 2)
    assert qh.tolist() == [1.0, None, 0.0]
    assert cycle._normalize_Q_heat(None, 2, 2).shape == (3,)
    with pytest.raises(ValueError):
        cycle._normalize_Q_heat(np.array([np.nan, 1.0]), 2, 2)
    with pytest.raises(ValueError):
        cycle._normalize_Q_heat(np.array([1.0, 2.0, 3.0, 4.0]), 2, 2)

    qcool_none = cycle._normalize_Q_cool(None, 2, 2)
    assert qcool_none.tolist() == [0.0, 0.0, None]
    qcool_scalar = cycle._normalize_Q_cool(np.array([3.0]), 2, 2)
    assert qcool_scalar.tolist() == [3.0, 3.0, 3.0]
    qcool_last_nan = cycle._normalize_Q_cool(
        np.array([1.0, 2.0, np.nan], dtype=object), 2, 2
    )
    assert qcool_last_nan.tolist() == [1.0, 2.0, None]
    with pytest.raises(ValueError):
        cycle._normalize_Q_cool(np.array([None, 1.0, 2.0], dtype=object), 2, 2)
    with pytest.raises(ValueError):
        cycle._normalize_Q_cool(np.array([1.0, np.nan, 2.0], dtype=object), 2, 2)
    with pytest.raises(ValueError):
        cycle._normalize_Q_cool(np.array([["x"]], dtype=object), 2, 2)

    inf = cycle._validate_T_cond_and_evap(
        np.array([100.0, 80.0]), np.array([40.0, 20.0])
    )
    assert inf >= 0.0


@pytest.mark.parametrize(
    "Q_heat, n_heat, n_cool, expected",
    [
        (100.0, 1, 1, [100.0]),
        (100.0, 1, 2, [100.0, 0.0]),
        (np.array([100.0]), 1, 2, [100.0, 0.0]),
        (100.0, 1, 3, [100.0, 0.0, 0.0]),
        (100.0, 2, 1, [100.0, 100.0]),
        (100.0, 2, 2, [100.0, 100.0, 0.0]),
        (np.nan, 1, 2, [None, 0.0]),
    ],
)
def test_cascade_scalar_q_heat_applies_to_each_process_condenser(
    Q_heat, n_heat, n_cool, expected
):
    hp = CascadeVapourCompressionCycle()

    assert hp._normalize_Q_heat(Q_heat, n_heat, n_cool).tolist() == expected


@pytest.mark.parametrize("Q_heat", [100.0, np.array([100.0]), np.array([100.0, 0.0])])
def test_cascade_scalar_q_heat_matches_the_explicit_stage_array(Q_heat):
    cycle = CascadeVapourCompressionCycle()
    cycle.solve(
        T_evap=np.array([24.0, 5.0]),
        T_cond=np.array([80.0]),
        dtcont=0.0,
        refrigerant="ammonia",
        Q_heat=Q_heat,
    )

    assert cycle.solved
    assert cycle.Q_heat_arr == pytest.approx([100.0, 0.0])
    assert cycle.Q_heat == pytest.approx(cycle.Q_cool + cycle.work)


def test_cascade_property_arrays_and_unsolved_work_branch():
    hp = CascadeVapourCompressionCycle()
    hp._subcycles = [_DummyCascadeCycle(), _DummyCascadeCycle()]
    hp._solved = True
    hp._dt_cascade_hx = 1.0

    assert hp.Q_evap_arr.shape == (2,)
    assert hp.Q_cas_cool_arr.shape == (2,)
    assert hp.Q_cool_arr.shape == (2,)
    assert hp.Q_cond_arr.shape == (2,)
    assert hp.Q_cas_heat_arr.shape == (2,)
    assert hp.Q_heat_arr.shape == (2,)
    assert hp.work_arr.shape == (2,)
    assert hp.dtcont == 0.0
    assert hp.dt_diff_max == 0.5

    hp._solved = False
    hp._max_work = 123.0
    assert hp.work == 123.0


def test_cascade_normalize_q_cool_and_numeric_conversion_errors():
    hp = CascadeVapourCompressionCycle()

    scalar = hp._normalize_Q_cool(5.0, n_heat=2, n_cool=2)
    assert scalar.shape == (3,)

    scalar_nan = hp._normalize_Q_cool(np.nan, n_heat=2, n_cool=2)
    assert scalar_nan.tolist() == [0.0, 0.0, None]

    with pytest.raises(ValueError, match="Incompatible Q_cool input"):
        hp._normalize_Q_cool([1.0, 2.0, 3.0, 4.0], n_heat=2, n_cool=2)

    with pytest.raises(ValueError, match="Q_cool values must be numeric"):
        hp._normalize_Q_cool([object(), None, None], n_heat=2, n_cool=2)

    with pytest.raises(ValueError, match="Q_cool values must be numeric"):
        hp._normalize_Q_cool([0.0, 0.0, object()], n_heat=2, n_cool=2)

    with pytest.raises(ValueError, match="Input must be numeric"):
        hp._as_1d_numeric_array("bad")


def test_cascade_normalize_q_heat_edge_cases_and_numeric_errors():
    hp = CascadeVapourCompressionCycle()

    assert hp._normalize_Q_heat(None, n_heat=2, n_cool=2).tolist() == [
        0.0,
        None,
        0.0,
    ]
    assert hp._normalize_Q_heat([None], n_heat=2, n_cool=2).tolist() == [
        0.0,
        None,
        0.0,
    ]
    assert hp._normalize_Q_heat([0.0, None, 5.0], n_heat=2, n_cool=2).tolist() == [
        0.0,
        None,
        5.0,
    ]

    with pytest.raises(ValueError, match="Incompatible Q_heat input"):
        hp._normalize_Q_heat(np.array([[1.0, 2.0]]), n_heat=2, n_cool=2)
    with pytest.raises(ValueError, match="Only the balancing stage's Q_heat"):
        hp._normalize_Q_heat([None, 1.0, 0.0], n_heat=2, n_cool=2)
    with pytest.raises(ValueError, match="Q_heat values must be numeric"):
        hp._normalize_Q_heat([0.0, object(), 0.0], n_heat=2, n_cool=2)


def test_cascade_allocate_process_duties_covers_heat_pump_and_refrigeration_modes():
    hp = CascadeVapourCompressionCycle()

    Q_heat, Q_cool = hp._allocate_process_duties(
        Q_heat=None,
        Q_cool=np.array([2.0]),
        duty_allocation=DutyAllocationRequest(
            heat=StageDutyRequest(
                Q_base=100.0,
                x_split=np.array([0.6, 1.0]),
                Q_available=np.array([50.0, 30.0]),
            ),
            cool=StageDutyRequest(
                Q_base=20.0, x_split=np.array([1.0]), Q_available=np.array([10.0])
            ),
        ),
        is_heat_pump=True,
    )

    # Each stage takes its fraction of its own availability.
    np.testing.assert_allclose(Q_heat, [30.0, 30.0])
    np.testing.assert_allclose(Q_cool, [10.0, np.nan])

    Q_heat, Q_cool = hp._allocate_process_duties(
        Q_heat=np.array([1.0]),
        Q_cool=None,
        duty_allocation=DutyAllocationRequest(
            heat=StageDutyRequest(
                Q_base=30.0, x_split=np.array([1.0]), Q_available=np.array([20.0])
            ),
            cool=StageDutyRequest(
                Q_base=80.0,
                x_split=np.array([0.5, 1.0]),
                Q_available=np.array([30.0, 30.0]),
            ),
        ),
        is_heat_pump=False,
    )

    np.testing.assert_allclose(Q_heat, [np.nan, 20.0])
    np.testing.assert_allclose(Q_cool, [15.0, 30.0])


def test_cascade_solve_invalid_temperature_with_nonnumeric_duty_uses_default_work():
    hp = CascadeVapourCompressionCycle()

    work = hp.solve(
        T_evap=np.array([60.0]),
        T_cond=np.array([50.0]),
        dtcont=0.0,
        Q_heat=object(),
        Q_cool=1.0,
        is_heat_pump=False,
    )

    assert hp.solved is False
    # Default work 1 x (1 + violation); the violation is the 10 K inversion
    # plus the 1 K cascade approach the condenser must keep.
    assert work == pytest.approx(1.0 * (1.0 + 10.0 + 1.0))


def test_cascade_solve_refrigerant_singleton_list_branch(monkeypatch):
    hp = CascadeVapourCompressionCycle()
    monkeypatch.setattr(hp, "_validate_T_cond_and_evap", lambda T_cond, T_evap: 0.0)
    monkeypatch.setattr(cascade_mod, "VapourCompressionCycle", _DummyCascadeCycle)

    work = hp.solve(
        T_evap=np.array([25.0, 15.0]),
        T_cond=np.array([70.0, 60.0]),
        dtcont=0.0,
        refrigerant=["water"],
        Q_heat=np.array([1.0, 1.0]),
        Q_cool=None,
        is_heat_pump=True,
    )

    assert isinstance(work, float)
    assert hp.solved is True


def test_cascade_solve_propagates_unsolved_child_cycle(monkeypatch):
    class _UnsolvedCycle:
        solved = False
        work = -5.0

        def solve(self, **kwargs):
            return self.work

    hp = CascadeVapourCompressionCycle()
    monkeypatch.setattr(hp, "_validate_T_cond_and_evap", lambda T_cond, T_evap: 0.0)
    monkeypatch.setattr(cascade_mod, "VapourCompressionCycle", _UnsolvedCycle)

    work = hp.solve(
        T_evap=np.array([20.0]),
        T_cond=np.array([80.0]),
        dtcont=0.0,
        Q_heat=np.array([100.0]),
        Q_cool=np.array([None], dtype=object),
    )

    assert hp.solved is False
    assert work > 0.0


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
    cycle = CascadeVapourCompressionCycle()
    with pytest.raises(ValueError):
        cycle.solve(
            T_evap=np.array([20.0, 0.0]),
            T_cond=np.array([80.0, 60.0]),
            dtcont=0.0,
            dt_cascade_hx=5.0,
            dT_superheat=np.array([5.0, 5.0, 5.0]),
            dT_subcool=np.array([2.0, 2.0, 2.0]),
            dT_ihx_gas_side=np.array([5.0, 5.0, 5.0]),
            eta_comp=0.75,
            refrigerant=["R134a", "R134a", "R134a"],
            Q_heat=np.array([1200.0, 700.0, 0.0]),
            Q_cool=bad_q_cool,
        )


def test_cascade_q_heat_allows_top_stage_to_be_unspecified_in_refrigeration():
    cycle = CascadeVapourCompressionCycle()
    cycle.solve(
        T_evap=np.array([20.0, 0.0]),
        T_cond=np.array([80.0, 60.0]),
        dtcont=0.0,
        dt_cascade_hx=5.0,
        dT_superheat=np.array([5.0, 5.0, 5.0]),
        dT_subcool=np.array([2.0, 2.0, 2.0]),
        dT_ihx_gas_side=np.array([5.0, 5.0, 5.0]),
        eta_comp=0.75,
        refrigerant=["R134a", "R134a", "R134a"],
        Q_heat=np.array([np.nan, 0.0]),
        Q_cool=np.array([0.0, 900.0]),
        is_heat_pump=False,
    )
    assert cycle.Q_heat_arr[0] == pytest.approx(cycle.Q_cond_arr[0])
    assert cycle.Q_heat_arr[1] == pytest.approx(0.0)
    assert cycle.Q_heat_arr[2] == pytest.approx(0.0)


def _two_stage_cascade(Q_heat, dT_subcool):
    cycle = CascadeVapourCompressionCycle()
    cycle.solve(
        T_evap=np.array([-0.2]),
        T_cond=np.array([84.9, 84.8]),
        dtcont=5.0,
        dT_subcool=np.array(dT_subcool),
        dT_ihx_gas_side=0.0,
        eta_comp=0.7,
        refrigerant=["Ammonia", "R1234ze(E)"],
        dt_cascade_hx=5.0,
        Q_heat=np.array(Q_heat),
        is_heat_pump=True,
    )
    assert cycle.solved
    return cycle


def test_subcooling_heat_does_not_reach_the_upper_evaporator():
    # All the lower stage's heat goes up to an evaporator at 79.8 C, so liquid
    # subcooled towards 0 C has nowhere to put the heat it gives up.
    plain = _two_stage_cascade([250.0, 0.0], [0.0, 0.0])
    subcooled = _two_stage_cascade([250.0, 0.0], [0.0, 85.0])

    assert subcooled.dT_subcool[1] == pytest.approx(0.0, abs=1e-6)
    assert subcooled.work == pytest.approx(plain.work, rel=1e-9)


def test_stage_passing_heat_up_can_subcool_into_its_process_duty():
    plain = _two_stage_cascade([200.0, 100.0], [0.0, 0.0])
    subcooled = _two_stage_cascade([200.0, 100.0], [0.0, 20.0])

    assert subcooled.dT_subcool[1] == pytest.approx(20.0)
    assert subcooled.work < plain.work
    stage = subcooled._subcycles[1]
    assert stage.Q_heat == pytest.approx(100.0)
    # The subcooling heat is delivered to the process, down to the liquid outlet.
    profile = stage._build_condenser_profile()
    assert profile[:, 1].min() == pytest.approx(64.8, abs=1e-6)


def test_process_duty_limits_subcooling_on_a_stage_passing_heat_up():
    # A small process duty can only take the coldest of the liquid's heat.
    stage = _two_stage_cascade([250.0, 5.0], [0.0, 85.0])._subcycles[1]

    assert 0.0 < stage.dT_subcool < 85.0
    # The cascade exchanger's share ends at the saturated liquid.
    from CoolProp.CoolProp import PropsSI

    saturated_liquid = PropsSI("H", "T", 84.8 + 273.15, "Q", 0, "R1234ze(E)")
    assert stage.Hs[4] == pytest.approx(saturated_liquid, rel=1e-4)


def test_stage_with_no_heat_to_pass_up_subcools_freely():
    stage = VapourCompressionCycle()
    stage.solve(
        T_evap=0.0,
        T_cond=80.0,
        dtcont=5.0,
        dT_subcool=40.0,
        eta_comp=0.7,
        refrigerant="Ammonia",
        dT_ihx_gas_side=0.0,
        Q_heat=100.0,
    )

    assert stage.dT_subcool == pytest.approx(40.0)


def test_ihx_limit_follows_the_subcooling_the_process_duty_allows():
    def stage(dT_subcool):
        cycle = VapourCompressionCycle()
        cycle.solve(
            T_evap=0.0,
            T_cond=80.0,
            dtcont=5.0,
            dT_subcool=dT_subcool,
            eta_comp=0.7,
            refrigerant="Ammonia",
            dT_ihx_gas_side=30.0,
            Q_heat=0.0,
            Q_cas_heat=100.0,
        )
        return cycle

    # No process duty, so the request cannot be met and must not limit the IHX.
    capped, plain = stage(80.0), stage(0.0)

    assert capped.dT_subcool == pytest.approx(0.0, abs=1e-6)
    assert capped.work == pytest.approx(plain.work, rel=1e-9)
    assert capped.penalty == pytest.approx(plain.penalty)


@pytest.mark.parametrize("Q_heat", [1.0, 5.0, 20.0, 60.0])
@pytest.mark.parametrize("dT_ihx", [5.0, 30.0, 60.0, 200.0])
def test_ihx_and_subcooling_settle_on_a_stage_with_a_small_process_duty(Q_heat, dT_ihx):
    stage = VapourCompressionCycle()
    stage.solve(
        T_evap=0.0,
        T_cond=80.0,
        dtcont=5.0,
        dT_subcool=78.0,
        eta_comp=0.7,
        refrigerant="R245fa",
        dT_ihx_gas_side=dT_ihx,
        Q_heat=Q_heat,
        Q_cas_heat=100.0,
    )

    assert stage.solved
    assert stage.dT_subcool < 78.0


def _two_condenser_refrigeration(Q_heat=None, dT_subcool=0.0):
    cycle = CascadeVapourCompressionCycle()
    cycle.solve(
        T_evap=np.array([-20.0]),
        T_cond=np.array([40.0, 10.0]),
        dtcont=5.0,
        dT_subcool=dT_subcool,
        dT_ihx_gas_side=0.0,
        eta_comp=0.7,
        refrigerant="Ammonia",
        dt_cascade_hx=5.0,
        Q_heat=Q_heat,
        Q_cool=np.array([100.0]),
        is_heat_pump=False,
    )
    assert cycle.solved
    return cycle


def test_refrigeration_cascade_passes_heat_from_cold_stage_to_warm_stage():
    cycle = _two_condenser_refrigeration()
    warm, cold = cycle.subcycles

    assert (warm.T_cond, warm.T_evap) == pytest.approx((40.0, 5.0))
    assert (cold.T_cond, cold.T_evap) == pytest.approx((10.0, -20.0))
    # The cold stage's condenser at 10 C feeds the warm stage's evaporator at 5 C.
    assert cold.Q_cas_heat == pytest.approx(cold.Q_cond)
    assert warm.Q_cas_cool == pytest.approx(cold.Q_cas_heat)
    assert warm.Q_cas_heat == pytest.approx(0.0)
    assert cold.Q_cas_cool == pytest.approx(0.0)
    # The warm stage rejects everything to the process at 40 C.
    assert warm.Q_heat == pytest.approx(warm.Q_cond)
    assert cold.Q_heat == pytest.approx(0.0)


def test_refrigeration_cascade_delivers_requested_cooling_once():
    cycle = _two_condenser_refrigeration()

    np.testing.assert_allclose(cycle.Q_cool_arr, [0.0, 100.0], atol=1e-9)


@pytest.mark.parametrize(
    ("T_cond", "T_evap", "Q_cool"),
    [
        ([40.0, 10.0], [-20.0], [100.0]),
        ([40.0], [5.0, -20.0], [30.0, 100.0]),
        ([60.0, 40.0], [5.0, -20.0], [30.0, 100.0]),
    ],
)
def test_refrigeration_cascade_conserves_energy(T_cond, T_evap, Q_cool):
    cycle = CascadeVapourCompressionCycle()
    cycle.solve(
        T_evap=np.array(T_evap),
        T_cond=np.array(T_cond),
        dtcont=5.0,
        dT_ihx_gas_side=0.0,
        eta_comp=0.7,
        refrigerant="Ammonia",
        dt_cascade_hx=5.0,
        Q_cool=np.array(Q_cool),
        is_heat_pump=False,
    )

    assert cycle.solved
    assert cycle.Q_cool == pytest.approx(sum(Q_cool))
    assert cycle.Q_heat == pytest.approx(cycle.Q_cool + cycle.work)


def test_refrigeration_cascade_matches_heat_pump_mode_at_the_same_duties():
    refrigeration = _two_condenser_refrigeration()

    heat_pump = CascadeVapourCompressionCycle()
    heat_pump.solve(
        T_evap=np.array([-20.0]),
        T_cond=np.array([40.0, 10.0]),
        dtcont=5.0,
        dT_ihx_gas_side=0.0,
        eta_comp=0.7,
        refrigerant="Ammonia",
        dt_cascade_hx=5.0,
        Q_heat=np.array([refrigeration.Q_heat, 0.0]),
        is_heat_pump=True,
    )

    assert refrigeration.work == pytest.approx(heat_pump.work)
    assert refrigeration.Q_cool == pytest.approx(heat_pump.Q_cool)


def test_refrigeration_stage_delivering_process_heat_passes_the_rest_up():
    warm, cold = _two_condenser_refrigeration(Q_heat=[np.nan, 50.0]).subcycles

    assert cold.Q_heat == pytest.approx(50.0)
    assert warm.Q_cas_cool == pytest.approx(cold.Q_cond - 50.0)
    assert warm.Q_heat == pytest.approx(warm.Q_cond)


def test_refrigeration_only_the_top_stage_may_leave_q_heat_unspecified():
    with pytest.raises(ValueError, match="Q_heat"):
        _two_condenser_refrigeration(Q_heat=[0.0, np.nan])


def test_refrigeration_subcooling_heat_does_not_reach_the_upper_evaporator():
    # All the cold stage's heat goes up to an evaporator at 5 C, so liquid
    # subcooled below 10 C has nowhere to put the heat it gives up.
    plain = _two_condenser_refrigeration()
    subcooled = _two_condenser_refrigeration(dT_subcool=np.array([0.0, 25.0]))

    assert subcooled.dT_subcool[1] == pytest.approx(0.0, abs=1e-6)
    assert subcooled.work == pytest.approx(plain.work, rel=1e-9)


def test_refrigeration_top_stage_subcools_into_the_process_freely():
    plain = _two_condenser_refrigeration()
    subcooled = _two_condenser_refrigeration(dT_subcool=np.array([10.0, 0.0]))

    assert subcooled.dT_subcool[0] == pytest.approx(10.0)
    assert subcooled.work < plain.work


def test_refrigeration_process_duty_limits_subcooling_on_a_stage_passing_heat_up():
    # A small process duty can only take the coldest of the liquid's heat.
    _, cold = _two_condenser_refrigeration(
        Q_heat=[np.nan, 5.0], dT_subcool=np.array([0.0, 25.0])
    ).subcycles

    assert 0.0 < cold.dT_subcool < 25.0
    # The cascade exchanger's share ends at the saturated liquid.
    from CoolProp.CoolProp import PropsSI

    saturated_liquid = PropsSI("H", "T", 10.0 + 273.15, "Q", 0, "Ammonia")
    assert cold.Hs[4] == pytest.approx(saturated_liquid, rel=1e-4)
    assert cold.Q_heat == pytest.approx(5.0)


def test_refrigeration_large_process_duty_leaves_subcooling_unlimited():
    _, cold = _two_condenser_refrigeration(
        Q_heat=[np.nan, 1e6], dT_subcool=np.array([0.0, 25.0])
    ).subcycles

    assert cold.dT_subcool == pytest.approx(25.0)
    assert cold.Q_cas_heat == pytest.approx(0.0)
