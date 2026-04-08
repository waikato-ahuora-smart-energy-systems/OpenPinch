"""Additional branch coverage tests for heat pump network wrappers."""

import numpy as np
import pytest

import OpenPinch.classes.cascade_heat_pump as cascade_mod
import OpenPinch.classes.multi_simple_heat_pump as multi_mod
from OpenPinch.classes.cascade_heat_pump import CascadeHeatPumpCycle
from OpenPinch.classes.multi_simple_heat_pump import MultiSimpleHeatPumpCycle


class _DummyCycle:
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


def test_cascade_property_arrays_and_unsolved_work_branch():
    hp = CascadeHeatPumpCycle()
    hp._subcycles = [_DummyCycle(), _DummyCycle()]
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
    hp = CascadeHeatPumpCycle()

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


def test_cascade_solve_refrigerant_singleton_list_branch(monkeypatch):
    hp = CascadeHeatPumpCycle()
    monkeypatch.setattr(hp, "_validate_T_cond_and_evap", lambda T_cond, T_evap: 0.0)
    monkeypatch.setattr(cascade_mod, "SimpleHeatPumpCycle", _DummyCycle)

    work = hp.solve(
        T_evap=np.array([25.0, 15.0]),
        T_cond=np.array([70.0, 60.0]),
        refrigerant=["water"],
        Q_heat=np.array([1.0, 1.0]),
        Q_cool=None,
    )

    assert isinstance(work, float)
    assert hp.solved is True


def test_multi_simple_property_arrays_and_helper_error_branches():
    hp = MultiSimpleHeatPumpCycle()
    hp._subcycles = [_DummyCycle(), _DummyCycle()]
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
    monkeypatch.setattr(multi_mod, "SimpleHeatPumpCycle", _DummyCycle)

    work = hp.solve(
        T_evap=np.array([20.0, 15.0]),
        T_cond=np.array([60.0, 55.0]),
        refrigerant=["water"],
        Q_heat=np.array([1.0, 2.0]),
        Q_cool=np.array([None, None]),
    )

    assert isinstance(work, float)
    assert hp.solved is True
