from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.services.heat_pump_integration.cycles import (
    brayton as hp_brayton,
)

from ..helpers import _base_args


def test_optimise_brayton_heat_pump_placement_is_explicitly_unsupported():
    args = SimpleNamespace(
        n_cond=1,
        n_evap=1,
        refrigerant_ls=["AIR"],
        T_cold=np.array([120.0, 80.0, 40.0]),
        H_cold=np.array([200.0, 100.0, 0.0]),
        T_hot=np.array([100.0, 70.0, 30.0]),
        H_hot=np.array([0.0, -80.0, -160.0]),
        dt_range_max=90.0,
        Q_hpr_target=200.0,
        eta_comp=0.75,
        eta_exp=0.75,
        dt_phase_change=1.0,
        dtcont_hp=5.0,
        dt_env_cont=5.0,
        T_env=20.0,
    )

    with pytest.raises(NotImplementedError, match="currently unsupported"):
        hp_brayton.optimise_brayton_heat_pump_placement(args)


def test_brayton_x0_and_bounds_shapes_are_consistent():
    args = SimpleNamespace(
        T_cold=np.array([140.0, 80.0, 20.0]),
        T_hot=np.array([130.0, 70.0, 10.0]),
        dt_range_max=130.0,
    )

    x0, bnds = hp_brayton._get_brayton_hp_opt_setup(args)

    assert len(x0) == 4
    assert len(bnds) == 4
    assert np.all(
        (np.array(x0) >= np.array([b[0] for b in bnds]))
        & (np.array(x0) <= np.array([b[1] for b in bnds]))
    )


def test_brayton_helper_paths(monkeypatch):
    args = _base_args(n_cond=1, n_evap=1)

    T_co, dT_c, dT_gc, q_h = hp_brayton._parse_brayton_hp_state_variables(
        np.array([0.1, 0.2, 0.3, 0.9]), args
    )
    assert len(T_co) == len(dT_c) == len(dT_gc) == len(q_h) == 1

    class _FakeBraytonCycle:
        def __init__(self):
            self.work_net = 15.0
            self.Q_cool = 40.0
            self.cycle_states = [{}, {}, {}, {"T": 50.0}]

        def solve(self, **kwargs):
            return None

    monkeypatch.setattr(hp_brayton, "SimpleBraytonHeatPumpCycle", _FakeBraytonCycle)
    hp_list = hp_brayton._create_brayton_hp_list(
        T_comp_out=np.array([120.0]),
        dT_gc=np.array([20.0]),
        Q_gc=np.array([100.0]),
        dT_comp=np.array([10.0]),
        args=args,
    )
    assert len(hp_list) == 1
