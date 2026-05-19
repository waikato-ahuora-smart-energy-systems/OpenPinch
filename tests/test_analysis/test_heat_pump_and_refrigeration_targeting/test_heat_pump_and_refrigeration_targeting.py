from types import SimpleNamespace

import numpy as np
import pytest

import OpenPinch.services.heat_pump_integration as hp_pkg
from OpenPinch.classes.problem_table import ProblemTable
from OpenPinch.classes.stream import Stream
from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.lib.enums import GT, PT, HPRcycle
from OpenPinch.lib.schemas.hpr import (
    HPRBackendResult,
    HPRThermoArtifacts,
)
from OpenPinch.services.common.graph_data import _create_graph_set
from OpenPinch.services.heat_pump_integration import (
    heat_pump_and_refrigeration_entry as hp,
)
from OpenPinch.services.heat_pump_integration.common import shared as hp_shared

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
            "T_col": np.array([120.0, 60.0]),
            "updates": {
                PT.H_NET_UT.value: np.array([0.0, 0.0]),
                PT.H_HOT_UT.value: np.array([0.0, 0.0]),
                PT.H_COLD_UT.value: np.array([0.0, 0.0]),
            },
        },
    )
    ambient_hot = np.array([-4.0, 1.0]) if q_amb_hot > 0.0 else np.zeros(2)
    ambient_cold = np.array([4.0, -1.0]) if q_amb_cold > 0.0 else np.zeros(2)
    monkeypatch.setattr(
        hp,
        "get_process_heat_cascade",
        lambda **_kwargs: _pt_with_hnet(
            4.0,
            -1.0,
            h_hot=ambient_hot,
            h_cold=ambient_cold,
        ),
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


def test_calc_hpr_cascade_uses_shared_temperature_intervals_for_hpr_and_air(
    monkeypatch,
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
        lambda streams, is_shifted=True: ProblemTable(
            {PT.T.value: [120.0, 100.0, 60.0]}
        ),
    )
    monkeypatch.setattr(
        hp,
        "get_process_heat_cascade",
        lambda **_kwargs: ProblemTable(
            {
                PT.T.value: [120.0, 90.0, 60.0],
                PT.H_NET.value: [6.0, 3.0, 0.0],
                PT.H_NET_HOT.value: [0.0, 3.0, 6.0],
                PT.H_NET_COLD.value: [6.0, 3.0, 0.0],
            }
        ),
    )
    monkeypatch.setattr(
        hp,
        "get_utility_heat_cascade",
        lambda **kwargs: {
            "T_col": np.array(kwargs["T_int_vals"], dtype=float),
            "updates": {
                PT.H_NET_UT.value: np.zeros(len(kwargs["T_int_vals"])),
                PT.H_HOT_UT.value: np.zeros(len(kwargs["T_int_vals"])),
                PT.H_COLD_UT.value: np.zeros(len(kwargs["T_int_vals"])),
            },
        },
    )

    amb_streams = StreamCollection()
    amb_streams.add(Stream(name="Air", t_supply=50.0, t_target=80.0, heat_flow=10.0))
    res = SimpleNamespace(
        hpr_hot_streams=StreamCollection(),
        hpr_cold_streams=StreamCollection(),
        Q_amb_hot=10.0,
        Q_amb_cold=0.0,
        amb_streams=amb_streams,
    )

    out = hp._calc_hpr_cascade(
        pt,
        res,
        is_T_vals_shifted=True,
        is_heat_pumping=True,
    )

    np.testing.assert_allclose(
        out.col[PT.T.value], np.array([120.0, 100.0, 90.0, 60.0])
    )
    np.testing.assert_allclose(
        out.col[PT.H_NET_W_AIR.value], np.array([7.0, 5.0, 4.0, 1.0])
    )
    np.testing.assert_allclose(
        out.col[PT.H_NET_HOT.value], np.array([2.0, 4.0, 5.0, 8.0])
    )
    np.testing.assert_allclose(
        out.col[PT.H_NET_COLD.value], np.array([9.0, 7.0, 6.0, 3.0])
    )


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

    assert shown["called"] is False
    assert figure.layout.title.text == "HP Profile"
    assert [trace.name for trace in figure.data] == [
        "Sink",
        "Source",
        "Condenser",
        "Evaporator",
    ]


def test_public_heat_pump_service_package_does_not_export_profile_helper():
    assert not hasattr(hp_pkg, "plot_multi_hp_profiles_from_results")


def test_direct_heat_pump_graph_payloads_include_nlp_and_hpr_overlay():
    pt = ProblemTable(
        {
            PT.T.value: [120.0, 80.0],
            PT.H_NET_HOT.value: [8.0, 2.0],
            PT.H_NET_COLD.value: [1.0, 6.0],
            PT.H_HOT_UT.value: [0.0, 3.0],
            PT.H_COLD_UT.value: [4.0, 0.0],
            PT.H_HOT_HP.value: [0.0, 5.0],
            PT.H_COLD_HP.value: [3.0, 0.0],
            PT.H_NET_W_AIR.value: [4.0, 1.0],
            PT.H_NET_HP.value: [2.0, -2.0],
        }
    )

    graphs = hp._get_hpr_graphs(pt, is_direct=True, is_heat_pumping=True)

    assert set(graphs) == {GT.NLP.value, GT.GCC_HP.value}
    assert list(graphs[GT.NLP.value].columns) == [
        PT.T.value,
        PT.H_NET_HOT.value,
        PT.H_NET_COLD.value,
        PT.H_HOT_UT.value,
        PT.H_COLD_UT.value,
        PT.H_HOT_HP.value,
        PT.H_COLD_HP.value,
    ]

    graph_set = _create_graph_set(
        SimpleNamespace(
            name="Direct Heat Pump",
            type="Direct Heat Pump",
            graphs=graphs,
        )
    )
    nlp_graph = next(
        graph for graph in graph_set["graphs"] if graph["type"] == GT.NLP.value
    )
    segment_titles = {segment["title"] for segment in nlp_graph["segments"]}
    assert "Heat Pump Condenser" in segment_titles
    assert "Heat Pump Evaporator" in segment_titles


@pytest.mark.parametrize(
    "hpr_type",
    [
        HPRcycle.MultiTempCarnot.value,
        HPRcycle.MultiSimpleCarnot.value,
        HPRcycle.MultiSimpleVapourComp.value,
        HPRcycle.CascadeVapourComp.value,
    ],
)
def test_get_hpr_targets_validates_supported_non_brayton_backend_results(
    monkeypatch, hpr_type
):
    monkeypatch.setattr(
        hp,
        "construct_HPRTargetInputs",
        lambda **kwargs: SimpleNamespace(hpr_type=hpr_type),
    )
    monkeypatch.setitem(
        hp._HP_PLACEMENT_HANDLERS,
        hpr_type,
        lambda args: HPRBackendResult(
            obj=0.1,
            utility_tot=1.0,
            w_net=0.5,
            Q_ext_heat=0.25,
            Q_ext_cold=0.25,
            Q_amb_hot=0.0,
            Q_amb_cold=0.0,
            artifacts=HPRThermoArtifacts(hpr_streams=StreamCollection()),
            amb_streams=StreamCollection(),
        ),
    )

    out = hp._get_hpr_targets(
        Q_hpr_target=10.0,
        T_vals=np.array([120.0, 80.0]),
        H_hot=np.array([0.0, -10.0]),
        H_cold=np.array([10.0, 0.0]),
        zone_config=SimpleNamespace(HPR_TYPE=hpr_type),
        is_heat_pumping=True,
    )

    assert isinstance(out, hp.HeatPumpTargetOutputs)
    assert out.success is True
    assert out.Q_ext == pytest.approx(0.5)


def test_hpr_handler_registry_includes_brayton():
    handler = hp._HP_PLACEMENT_HANDLERS[HPRcycle.Brayton.value]
    assert handler is hp.optimise_brayton_heat_pump_placement
