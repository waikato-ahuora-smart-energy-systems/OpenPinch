from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.analysis.heat_pump_and_refrigeration_placement import (
    brayton as hp_brayton,
)
from OpenPinch.classes.stream_collection import StreamCollection

from ..helpers import (
    _base_args,
    _patch_output_model_validate,
    _pt_with_hnet,
    _sc,
    _stream,
)


def test_optimise_brayton_heat_pump_placement_raises_on_failed_solver(monkeypatch):
    class DummyResult:
        success = False
        message = "forced failure"

    monkeypatch.setattr(hp_brayton, "minimize", lambda *args, **kwargs: DummyResult())

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
        Q_amb_max=0.0,
        dtcont_hp=5.0,
        dt_env_cont=5.0,
        T_env=20.0,
    )

    with pytest.raises(ValueError, match="Brayton heat pump targeting failed"):
        hp_brayton.optimise_brayton_heat_pump_placement(args)


def test_brayton_x0_and_bounds_shapes_are_consistent():
    args = SimpleNamespace(
        T_cold=np.array([140.0, 80.0, 20.0]),
        T_hot=np.array([130.0, 70.0, 10.0]),
        dt_range_max=130.0,
    )

    x0 = hp_brayton._get_x0_for_brayton_hp_opt(args)
    bnds = hp_brayton._get_bounds_for_brayton_hp_opt(args)

    assert len(x0) == 4
    assert len(bnds) == 4
    assert np.all(
        (np.array(x0) >= np.array([b[0] for b in bnds]))
        & (np.array(x0) <= np.array([b[1] for b in bnds]))
    )


def test_brayton_paths_and_helpers(monkeypatch):
    args = _base_args(n_cond=1, n_evap=1)
    _patch_output_model_validate(monkeypatch)

    class _OptRes:
        success = True
        x = np.array([0.1, 0.2, 0.3, 0.9])
        message = "ok"

    monkeypatch.setattr(hp_brayton, "minimize", lambda **kwargs: _OptRes())
    monkeypatch.setattr(
        hp_brayton,
        "_compute_brayton_hp_system_obj",
        lambda x, args: {
            "obj": 0.1,
            "utility_tot": 1.0,
            "net_work": 0.5,
            "Q_ext": 0.0,
            "Q_amb_hot": 0.0,
            "Q_amb_cold": 0.0,
            "cop_h": 2.0,
            "hot_streams": StreamCollection(),
            "cold_streams": StreamCollection(),
        },
    )
    out = hp_brayton.optimise_brayton_heat_pump_placement(args)
    assert out["success"] is True

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

    class _FakeHPObj:
        def __init__(self):
            self.work_net = 12.0
            self.Q_cool = 35.0
            self.cycle_states = [{}, {}, {}, {"T": 42.0}]

    monkeypatch.setattr(
        hp_brayton, "_create_brayton_hp_list", lambda **kwargs: [_FakeHPObj()]
    )
    monkeypatch.setattr(
        hp_brayton,
        "_build_simulated_hpr_streams",
        lambda hp_list, **kwargs: _sc(
            _stream("S", 100.0, 90.0, 10.0, is_process_stream=False)
        ),
    )
    seq = iter([_pt_with_hnet(8.0, -1.0), _pt_with_hnet(7.0, -2.0)])
    monkeypatch.setattr(
        hp_brayton, "get_process_heat_cascade", lambda **kwargs: next(seq)
    )

    out_perf = hp_brayton._compute_brayton_hp_system_obj(
        np.array([0.1, 0.2, 0.3, 0.9]), args
    )
    assert "cop_h" in out_perf
