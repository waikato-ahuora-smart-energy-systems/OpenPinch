"""Compressor isentropic efficiency correlations copied from HeatPumpLab."""

import pickle

import pytest

from OpenPinch.analysis.heat_pumps.common.compressor_efficiency import (
    CompressorConditions,
    brendel_efficiency,
    compressor_efficiency_model,
    pareto_front_efficiency,
    wang_cleland_efficiency,
)


@pytest.mark.parametrize(
    ("pressure_ratio", "expected"),
    [
        # HeatPumpLab 0.2.1, HeatPumpLab.components.compressor.pareto_front_efficiency
        (1.0, 0.8694998881727463),
        (2.0, 0.8547136696099422),
        (4.0, 0.8282634062808621),
        (8.0, 0.7910654021728641),
        (15.0, 0.6928732762854044),
        (20.0, 0.5783241094079505),
    ],
)
def test_pareto_front_matches_heatpumplab(pressure_ratio, expected):
    assert pareto_front_efficiency(pressure_ratio) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("pressure_ratio", "expected"),
    [(2.0, 0.7020833333333334), (4.0, 0.7145833333333333)],
)
def test_wang_cleland_is_the_heatpumplab_polynomial(pressure_ratio, expected):
    assert wang_cleland_efficiency(pressure_ratio) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("pressure_ratio", "suction_pressure_kpa", "expected"),
    [
        (3.0, 500.0, 0.6563042407638459),
        (5.0, 1000.0, 0.6513272645543872),
        # HeatPumpLab returns 0.4 when there is no compression.
        (1.0, 500.0, 0.4),
    ],
)
def test_brendel_takes_suction_pressure_in_kpa(
    pressure_ratio, suction_pressure_kpa, expected
):
    assert brendel_efficiency(pressure_ratio, suction_pressure_kpa) == pytest.approx(
        expected, rel=1e-12
    )


def _conditions(*, P_suction=500.0, P_discharge=2000.0):
    return CompressorConditions(
        refrigerant="AMMONIA",
        P_suction=P_suction,
        P_discharge=P_discharge,
        T_suction=10.0,
        T_evap_dew=5.0,
        T_cond_dew=50.0,
        rho_suction=3.5,
        rho_evap_dew=3.6,
        rho_cond_dew=15.0,
    )


def test_conditions_report_the_pressure_ratio():
    assert _conditions(P_suction=400.0, P_discharge=1800.0).pressure_ratio == 4.5


def test_constant_model_is_the_constant_efficiency():
    assert compressor_efficiency_model("constant", 0.7) == 0.7


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("pareto_front", 0.8282634062808621),
        ("wang_cleland", 0.7145833333333333),
        ("brendel", brendel_efficiency(4.0, 500.0)),
    ],
)
def test_named_models_read_the_compressor_conditions(name, expected):
    model = compressor_efficiency_model(name, 0.7)
    assert model(_conditions()) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("name", ["pareto_front", "wang_cleland", "brendel"])
def test_named_models_are_clipped_like_heatpumplab(name):
    model = compressor_efficiency_model(name, 0.7)
    assert model(_conditions(P_suction=10.0, P_discharge=500.0)) == 0.4


def test_a_function_is_used_as_given():
    def flat(conditions):
        return 0.65

    assert compressor_efficiency_model(flat, 0.7) is flat


@pytest.mark.parametrize("name", ["pareto_front", "wang_cleland", "brendel"])
def test_named_models_pickle_for_worker_processes(name):
    model = compressor_efficiency_model(name, 0.7)
    restored = pickle.loads(pickle.dumps(model))
    assert restored(_conditions()) == model(_conditions())
