from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.services.heat_pump_integration import (
    heat_pump_and_refrigeration_entry as hp,
)
from OpenPinch.services.heat_pump_integration.common import shared as hp_shared
from OpenPinch.classes.stream import Stream
from OpenPinch.classes.problem_table import ProblemTable
from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.lib.enums import PT

from .helpers import _base_args, _pt_with_hnet


@pytest.mark.parametrize(
    ("q_amb_hot", "q_amb_cold", "expected_hot", "expected_cold", "expected_w_air"),
    [
        (
            20.0,
            0.0,
            np.array([-2.0, 3.0]),
            np.array([3.0, 3.0]),
            np.array([5.0, 0.0]),
        ),
        (
            0.0,
            20.0,
            np.array([2.0, 2.0]),
            np.array([7.0, 2.0]),
            np.array([5.0, 0.0]),
        ),
        (
            0.0,
            0.0,
            np.array([2.0, 2.0]),
            np.array([3.0, 3.0]),
            np.array([1.0, 1.0]),
        ),
    ],
)
def test_calc_heat_pump_and_refrigeration_cascade_branches(
    monkeypatch, q_amb_hot, q_amb_cold, expected_hot, expected_cold, expected_w_air
):
    pt = ProblemTable(
        {
            PT.T.value: [120.0, 60.0],
            PT.H_NET_A.value: [1.0, 1.0],
            PT.H_NET_HOT.value: [2.0, 2.0],
            PT.H_NET_COLD.value: [3.0, 3.0],
        }
    )

    monkeypatch.setattr(
        hp,
        "create_problem_table_with_t_int",
        lambda streams, is_shifted=True: ProblemTable({PT.T.value: [120.0, 60.0]}),
    )
    monkeypatch.setattr(
        hp,
        "get_utility_heat_cascade",
        lambda **_kwargs: {
            PT.H_NET_UT.value: np.array([0.0, 0.0]),
            PT.H_HOT_UT.value: np.array([0.0, 0.0]),
            PT.H_COLD_UT.value: np.array([0.0, 0.0]),
        },
    )
    monkeypatch.setattr(
        hp, "get_process_heat_cascade", lambda **_kwargs: _pt_with_hnet(4.0, -1.0)
    )

    res = SimpleNamespace(
        hpr_hot_streams=StreamCollection(),
        hpr_cold_streams=StreamCollection(),
        Q_amb_hot=q_amb_hot,
        Q_amb_cold=q_amb_cold,
        amb_streams=hp_shared.get_ambient_air_stream(
            q_amb_hot, q_amb_cold, _base_args()
        ),
    )
    out = hp._calc_hpr_cascade(
        pt,
        res,
        is_T_vals_shifted=True,
        is_heat_pumping=True,
    )
    assert isinstance(out, ProblemTable)
    np.testing.assert_allclose(out.col[PT.H_NET_HOT.value], expected_hot)
    np.testing.assert_allclose(out.col[PT.H_NET_COLD.value], expected_cold)
    np.testing.assert_allclose(out.col[PT.H_NET_W_AIR.value], expected_w_air)


def test_plot_multi_hp_profiles_from_results_returns_plotly_figure(monkeypatch):
    shown = {"called": False}
    monkeypatch.setattr(
        hp_shared.go.Figure,
        "show",
        lambda self: shown.__setitem__("called", True),
    )

    hot_streams = StreamCollection()
    hot_streams.add(
        Stream(name="HP_H1", t_supply=110.0, t_target=100.0, heat_flow=20.0)
    )
    cold_streams = StreamCollection()
    cold_streams.add(Stream(name="HP_C1", t_supply=70.0, t_target=80.0, heat_flow=15.0))

    figure = hp_shared.plot_multi_hp_profiles_from_results(
        T_hot=np.array([120.0, 100.0]),
        H_hot=np.array([0.0, 20.0]),
        T_cold=np.array([80.0, 60.0]),
        H_cold=np.array([0.0, 15.0]),
        hpr_hot_streams=hot_streams,
        hpr_cold_streams=cold_streams,
        title="HP Profile",
    )

    assert shown["called"] is True
    assert figure.layout.title.text == "HP Profile"
    assert [trace.name for trace in figure.data] == [
        "Sink",
        "Source",
        "Condenser",
        "Evaporator",
    ]
