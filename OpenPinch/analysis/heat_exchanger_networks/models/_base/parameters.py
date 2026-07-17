"""Parameter loading and operating-state normalization for base HEN models."""

from __future__ import annotations

import numpy as np


def set_blank_input_parameters(model) -> None:
    """Initialize the solver-array attributes expected by source equations."""

    model.T_h_in = np.array([], dtype=float)
    model.T_h_out = np.array([], dtype=float)
    model.f_h = np.array([], dtype=float)
    model.htc_h = np.array([], dtype=float)
    model.h_cost = np.array([], dtype=float)
    model.hot_names = np.array([], dtype=str)
    model.T_h_cont = np.array([], dtype=float)

    model.T_c_in = np.array([], dtype=float)
    model.T_c_out = np.array([], dtype=float)
    model.f_c = np.array([], dtype=float)
    model.htc_c = np.array([], dtype=float)
    model.c_cost = np.array([], dtype=float)
    model.cold_names = np.array([], dtype=str)
    model.T_c_cont = np.array([], dtype=float)

    model.hu_cost = np.array([], dtype=float)
    model.hu_unit_cost = np.array([], dtype=float)
    model.hu_coeff = np.array([], dtype=float)
    model.T_hu_in = np.array([], dtype=float)
    model.T_hu_out = np.array([], dtype=float)
    model.T_hu_cont = np.array([], dtype=float)
    model.htc_hu = np.array([], dtype=float)
    model.hu_exp = np.array([], dtype=float)

    model.cu_cost = np.array([], dtype=float)
    model.cu_unit_cost = np.array([], dtype=float)
    model.cu_coeff = np.array([], dtype=float)
    model.T_cu_in = np.array([], dtype=float)
    model.T_cu_out = np.array([], dtype=float)
    model.T_cu_cont = np.array([], dtype=float)
    model.htc_cu = np.array([], dtype=float)
    model.cu_exp = np.array([], dtype=float)

    model.unit_cost = np.array([], dtype=float)
    model.A_coeff = np.array([], dtype=float)
    model.A_exp = np.array([], dtype=float)
    model.period_ids = np.array(["0"], dtype=str)
    model.period_weights = np.array([1.0], dtype=float)
    model.N_periods = 1
    model.period_weight_sum = 1.0


def get_model_parameters_from_solver_arrays(model) -> None:
    """Populate model attributes from the OpenPinch private array adapter."""

    for name, values in model.solver_arrays.arrays.items():
        setattr(model, name, np.array(values, copy=True))
    model._normalise_state_arrays()
    model._set_minimum_approach_temperatures()


def _normalise_state_arrays(model) -> None:
    """Validate the explicit operating-period axis used by HEN models."""

    if "period_ids" not in model.solver_arrays.arrays:
        raise ValueError("period_ids is required for HEN model setup.")
    if "period_weights" not in model.solver_arrays.arrays:
        raise ValueError("period_weights is required for HEN model setup.")

    model.period_ids = np.asarray(model.period_ids, dtype=str)
    model.period_weights = np.asarray(model.period_weights, dtype=float)
    model.N_periods = int(len(model.period_ids))
    if model.N_periods <= 0:
        raise ValueError("HEN model construction requires at least one state.")
    if len(model.period_weights) != model.N_periods:
        raise ValueError(
            "HeatExchangerNetworkLabel period weight count must match period_id count."
        )
    if not np.isfinite(model.period_weights).all():
        raise ValueError("HeatExchangerNetworkLabel period weights must be finite.")
    model.period_weight_sum = float(np.sum(model.period_weights))
    if model.period_weight_sum <= 0.0:
        raise ValueError(
            "HeatExchangerNetworkLabel period weights must have a positive sum."
        )

    for base_name in (
        "T_h_in",
        "T_h_out",
        "f_h",
        "htc_h",
        "h_cost",
        "T_h_cont",
        "T_c_in",
        "T_c_out",
        "f_c",
        "htc_c",
        "c_cost",
        "T_c_cont",
        "T_hu_in",
        "T_hu_out",
        "htc_hu",
        "hu_cost",
        "T_hu_cont",
        "T_cu_in",
        "T_cu_out",
        "htc_cu",
        "cu_cost",
        "T_cu_cont",
    ):
        period_name = f"{base_name}_period"
        values = np.asarray(getattr(model, period_name, []), dtype=float)
        if values.size == 0:
            raise ValueError(f"{period_name} is required for HEN model setup.")
        if values.ndim != 2:
            raise ValueError(f"{period_name} must be indexed by operating period.")
        if values.shape[0] != model.N_periods:
            raise ValueError(
                f"{period_name} has {values.shape[0]} state rows; "
                f"expected {model.N_periods}."
            )
        setattr(model, period_name, values)
        setattr(model, base_name, values[0].copy())
