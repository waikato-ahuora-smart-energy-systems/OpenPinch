"""Additional branch tests for cascade and multi-cycle heat pump containers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.classes.cascade_heat_pump import CascadeHeatPumpCycle
from OpenPinch.classes.multi_simple_heat_pump import MultiSimpleHeatPumpCycle
from OpenPinch.classes.stream import Stream
from OpenPinch.classes.stream_collection import StreamCollection


def _stream_collection():
    sc = StreamCollection()
    sc.add(
        Stream(
            name="S",
            t_supply=100.0,
            t_target=90.0,
            heat_flow=10.0,
            is_process_stream=False,
        )
    )
    return sc


def _fake_cycle(
    *,
    solved: bool = True,
    refrigerant: str = "R134A",
    T_evap: float = 20.0,
    T_cond: float = 80.0,
    dT_superheat: float = 5.0,
    dT_subcool: float = 2.0,
    eta_comp: float = 0.75,
    dt_ihx_gas_side: float = 5.0,
    Q_evap: float = 60.0,
    Q_cond: float = 100.0,
    Q_cool: float = 55.0,
    Q_heat: float = 95.0,
    Q_cas_cool: float = 5.0,
    Q_cas_heat: float = 6.0,
    work: float = 40.0,
    penalty: float = 1.0,
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
        build_stream_collection=lambda **_kwargs: _stream_collection(),
    )


def test_cascade_property_sweep_and_normalization_branches():
    cycle = CascadeHeatPumpCycle()
    c1 = _fake_cycle()
    c2 = _fake_cycle(T_evap=10.0, T_cond=70.0, work=20.0, Q_evap=30.0, Q_cond=50.0)
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
    assert cycle.dt_ihx_gas_side.tolist() == [5.0, 5.0]
    assert cycle.num_cycles == 1
    assert len(cycle.subcycles) == 2

    cycle._solved = False
    cycle._max_work = 123.0
    assert cycle.work == 123.0
    assert cycle.penalty == 0

    cycle._solved = True
    cycle._subcycles = [_fake_cycle(work=0.0), _fake_cycle(work=0.0)]
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

    qh = cycle._normalize_Q_heat(np.array([np.nan, np.nan]), 2, 2)
    assert qh.tolist() == [1.0, 0.0, 0.0]
    assert cycle._normalize_Q_heat(None, 2, 2).shape == (3,)
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


def test_multi_simple_property_sweep_and_normalization_branches():
    cycle = MultiSimpleHeatPumpCycle()
    c1 = _fake_cycle()
    c2 = _fake_cycle(T_evap=15.0, T_cond=75.0, work=30.0, Q_evap=50.0, Q_cond=90.0)
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

    cycle._subcycles = [_fake_cycle(work=0.0), _fake_cycle(work=0.0)]
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
