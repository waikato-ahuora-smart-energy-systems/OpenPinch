"""The add-move dQ/dA a StageWise model ranks new matches by."""

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.models._stagewise.postprocess import (
    period_weighted_dqda,
)


def _owner(weights, hot, cold, theta_1, theta_2, u):
    """One hot and one cold stream in one stage, one value per period."""

    return SimpleNamespace(
        I=1,
        J=1,
        S=1,
        N_periods=len(weights),
        period_weights=weights,
        T_h_by_period=[[[h, h - 40.0]] for h in hot],
        T_c_by_period=[[[c + 30.0, c]] for c in cold],
        theta_1_by_period=[[[[t]]] for t in theta_1],
        theta_2_by_period=[[[[t]]] for t in theta_2],
        U_r_period=[[[value]] for value in u],
    )


def test_dqda_weights_every_period_by_its_weight():
    owner = _owner(
        weights=[3.0, 1.0],
        hot=[200.0, 180.0],
        cold=[100.0, 120.0],
        theta_1=[40.0, 20.0],
        theta_2=[30.0, 10.0],
        u=[0.5, 0.4],
    )
    q_r = [[[[100.0]]], [[[50.0]]]]

    first = 40.0 * 30.0 * 0.5 / (200.0 - 100.0)
    second = 20.0 * 10.0 * 0.4 / (180.0 - 120.0)
    assert period_weighted_dqda(owner, q_r)[0][0][0] == pytest.approx(
        (3.0 * first + 1.0 * second) / 4.0, rel=1e-12
    )


def test_dqda_of_one_period_is_that_period_s():
    owner = _owner(
        weights=[1.0],
        hot=[200.0],
        cold=[100.0],
        theta_1=[40.0],
        theta_2=[30.0],
        u=[0.5],
    )

    assert period_weighted_dqda(owner, [[[[100.0]]]])[0][0][0] == pytest.approx(
        40.0 * 30.0 * 0.5 / 100.0, rel=1e-12
    )


def test_dqda_of_a_period_without_duty_is_its_open_driving_force():
    owner = _owner(
        weights=[1.0, 1.0],
        hot=[200.0, 180.0],
        cold=[100.0, 120.0],
        theta_1=[40.0, 20.0],
        theta_2=[30.0, 10.0],
        u=[0.5, 0.4],
    )
    q_r = [[[[100.0]]], [[[0.0]]]]

    first = 40.0 * 30.0 * 0.5 / 100.0
    second = 0.4 * (180.0 - 120.0)
    assert period_weighted_dqda(owner, q_r)[0][0][0] == pytest.approx(
        (first + second) / 2.0, rel=1e-12
    )
