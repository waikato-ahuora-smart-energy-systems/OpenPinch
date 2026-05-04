"""Shared builders for heat-pump and refrigeration targeting tests."""

from types import SimpleNamespace

import numpy as np

from OpenPinch.services.heat_pump_integration import (
    heat_pump_and_refrigeration_entry as hp,
)
from OpenPinch.classes.problem_table import ProblemTable
from OpenPinch.classes.stream import Stream
from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.lib.enums import HPRcycle, PT


def _build_multi_temperature_profiles(
    T_cond, Q_cond, T_evap, Q_evap, *, eta_hp=0.5, eta_he=0.0
):
    T_cond = np.asarray(T_cond, dtype=float)
    Q_cond = np.asarray(Q_cond, dtype=float)
    T_evap = np.asarray(T_evap, dtype=float)
    Q_evap = np.asarray(Q_evap, dtype=float)

    T_cold = np.concatenate((T_cond, np.array([T_cond[-1] - 40.0])))
    H_cold = np.concatenate((np.flip(np.cumsum(np.flip(Q_cond))), np.array([0.0])))
    T_hot = np.concatenate((np.array([T_evap[0] + 40.0]), T_evap))
    H_hot = np.concatenate((np.array([0.0]), -np.cumsum(Q_evap)))
    args = SimpleNamespace(
        T_cold=T_cold,
        T_hot=T_hot,
        eta_ii_hpr_carnot=eta_hp,
        eta_ii_he_carnot=eta_he,
    )
    return args, H_hot, H_cold


def _sc(*streams):
    coll = StreamCollection()
    for stream in streams:
        coll.add(stream)
    return coll


def _stream(
    name: str,
    t_supply: float,
    t_target: float,
    heat_flow: float,
    *,
    is_process_stream: bool = True,
):
    return Stream(
        name=name,
        t_supply=t_supply,
        t_target=t_target,
        heat_flow=heat_flow,
        dt_cont=0.0,
        htc=1.0,
        is_process_stream=is_process_stream,
    )


def _base_args(**overrides):
    args = {
        "hpr_type": HPRcycle.MultiTempCarnot.value,
        "Q_hpr_target": 200.0,
        "dt_range_max": 110.0,
        "T_hot": np.array([140.0, 90.0, 40.0]),
        "H_hot": np.array([0.0, -80.0, -160.0]),
        "T_cold": np.array([130.0, 80.0, 30.0]),
        "H_cold": np.array([200.0, 100.0, 0.0]),
        "z_amb_hot": np.zeros(3),
        "z_amb_cold": np.zeros(3),
        "n_cond": 2,
        "n_evap": 2,
        "eta_comp": 0.75,
        "eta_exp": 0.7,
        "dtcont_hp": 5.0,
        "dt_hp_ihx": 3.0,
        "dt_cascade_hx": 2.0,
        "dt_phase_change": 1.0,
        "heat_to_power_ratio": 1.0,
        "cold_to_power_ratio": 0.0,
        "is_heat_pumping": True,
        "max_multi_start": 2,
        "T_env": 20.0,
        "dt_env_cont": 5.0,
        "eta_ii_hpr_carnot": 0.6,
        "eta_ii_he_carnot": 0.4,
        "refrigerant_ls": ["R134A", "R134A", "R134A"],
        "do_refrigerant_sort": False,
        "initialise_simulated_cycle": True,
        "allow_integrated_expander": True,
        "dT_subcool": None,
        "dT_superheat": None,
        "bckgrd_hot_streams": _sc(
            Stream(name="H", t_supply=120.0, t_target=80.0, heat_flow=50.0)
        ),
        "bckgrd_cold_streams": _sc(
            Stream(name="C", t_supply=30.0, t_target=60.0, heat_flow=40.0)
        ),
        "bb_minimiser": "rbf",
        "eta_penalty": 0.001,
        "rho_penalty": 10.0,
        "debug": False,
    }
    args.update(overrides)
    args.setdefault("Q_heat_max", float(args["H_cold"][0]))
    args.setdefault("Q_cool_max", float(-args["H_hot"][-1]))
    return SimpleNamespace(**args)


def _pt_with_hnet(h0, h1):
    return ProblemTable({PT.T.value: [120.0, 60.0], PT.H_NET.value: [h0, h1]})


def _patch_output_model_validate(monkeypatch):
    monkeypatch.setattr(
        hp.HPRTargetOutputs,
        "model_validate",
        classmethod(lambda cls, value: value),
    )
