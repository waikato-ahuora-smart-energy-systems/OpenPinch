"""A cascade heat pump's search also starts from each condenser alone.

At the default budget a 2x1 water cascade on ``basic_pinch`` came out at
256,246, 273,996 and 271,099 $/y for seeds 0, 1 and 2 (review finding 23):
the cheapest design puts all the heat into one condenser, and only some
seeds found it from the Carnot warm start. Starting also from the warm
start with all the heat in each condenser in turn, seeds 0 to 2 gave
267,803 to 268,454 $/y.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import OpenPinch.analysis.heat_pumps.targeting.cascade_vapour_compression as hp_cascade

from .helpers import _base_args


def _layout() -> hp_cascade.HPRoptVectorLayout:
    return hp_cascade.HPRoptVectorLayout(
        n_cond=2, n_evap=1, n_subcool=2, n_heat_split=2, n_cool_split=0, n_ihx=2
    )


def _starts(monkeypatch, args, seed) -> np.ndarray:
    """The starting points ``optimise_cascade_heat_pump_placement`` searches
    from, with ``seed`` as the Carnot warm start."""
    monkeypatch.setattr(hp_cascade, "initialise_hpr_seed", lambda *_a, **_k: seed)
    monkeypatch.setattr(hp_cascade, "preflight_coolprop_fluid_names", lambda *_a: None)
    monkeypatch.setattr(
        hp_cascade, "preflight_coolprop_hpr_targeting", lambda **_k: None
    )
    monkeypatch.setattr(
        hp_cascade,
        "validate_vapour_hp_refrigerant_ls",
        lambda num_stages, args: ["R134A"] * num_stages,
    )
    captured = {}

    def fake_solve_hpr_placement(*, f_obj, x0_ls, bnds, args):
        captured["x0_ls"] = np.asarray(x0_ls, dtype=float)
        return SimpleNamespace(success=True)

    monkeypatch.setattr(hp_cascade, "solve_hpr_placement", fake_solve_hpr_placement)
    hp_cascade.optimise_cascade_heat_pump_placement(args)
    return captured["x0_ls"]


def test_a_refrigerator_keeps_its_one_warm_start(monkeypatch):
    args = _base_args(n_cond=3, n_evap=1, is_heat_pumping=False)
    seed = SimpleNamespace(
        T_cond=np.array([125.0, 100.0, 80.0]),
        Q_cond=np.array([90.0, 70.0, 60.0]),
        T_evap=np.array([35.0]),
        Q_evap=np.array([150.0]),
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
    )

    assert _starts(monkeypatch, args, seed).ndim == 1


def test_a_cascade_search_starts_from_each_condenser_alone(monkeypatch):
    args = _base_args(n_cond=2, n_evap=1)
    seed = SimpleNamespace(
        T_cond=np.array([125.0, 100.0]),
        Q_cond=np.array([120.0, 80.0]),
        T_evap=np.array([35.0]),
        Q_evap=np.array([150.0]),
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
    )
    monkeypatch.setattr(hp_cascade, "initialise_hpr_seed", lambda *_a, **_k: seed)
    monkeypatch.setattr(hp_cascade, "preflight_coolprop_fluid_names", lambda *_a: None)
    monkeypatch.setattr(
        hp_cascade, "preflight_coolprop_hpr_targeting", lambda **_k: None
    )
    monkeypatch.setattr(
        hp_cascade,
        "validate_vapour_hp_refrigerant_ls",
        lambda num_stages, args: ["R134A"] * num_stages,
    )
    captured = {}

    def fake_solve_hpr_placement(*, f_obj, x0_ls, bnds, args):
        captured["x0_ls"] = np.asarray(x0_ls, dtype=float)
        return SimpleNamespace(success=True)

    monkeypatch.setattr(hp_cascade, "solve_hpr_placement", fake_solve_hpr_placement)

    hp_cascade.optimise_cascade_heat_pump_placement(args)

    starts = captured["x0_ls"]
    split = _layout().heat_split_slice
    assert starts.shape[0] == 3
    np.testing.assert_allclose(starts[1, split], [1.0, 0.0])
    np.testing.assert_allclose(starts[2, split], [0.0, 1.0])
    others = np.ones(starts.shape[1], dtype=bool)
    others[split] = False
    np.testing.assert_allclose(starts[1:, others], starts[[0, 0]][:, others])
