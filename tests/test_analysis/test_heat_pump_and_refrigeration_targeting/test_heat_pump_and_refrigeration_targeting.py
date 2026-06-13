from types import SimpleNamespace

import numpy as np
import pytest

import OpenPinch.services.heat_pump_integration as hp_pkg
from OpenPinch.classes.problem_table import ProblemTable
from OpenPinch.classes.stream import Stream
from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.classes.zone import Zone
from OpenPinch.lib.config import Configuration
from OpenPinch.lib.enums import GT, PT, TT, ZT, HPRcycle
from OpenPinch.lib.schemas.hpr import (
    HPRBackendResult,
    HPRThermoArtifacts,
)
from OpenPinch.services.common.graph_data import _create_graph_set
from OpenPinch.services.heat_pump_integration import (
    heat_pump_and_refrigeration_entry as hp,
)
from OpenPinch.services.heat_pump_integration.common import shared as hp_shared

from .helpers import _base_args, _patch_output_model_validate, _pt_with_hnet


def _make_base_utility_collections(
    *,
    hot_flow: float = 40.0,
    cold_flow: float = 30.0,
) -> tuple[StreamCollection, StreamCollection]:
    hot_utilities = StreamCollection()
    hot_utilities.add(
        Stream(
            name="Steam",
            t_supply=120.0,
            t_target=80.0,
            heat_flow=hot_flow,
            dt_cont=0.0,
        )
    )
    cold_utilities = StreamCollection()
    cold_utilities.add(
        Stream(
            name="Cooling Water",
            t_supply=20.0,
            t_target=60.0,
            heat_flow=cold_flow,
            dt_cont=0.0,
        )
    )
    return hot_utilities, cold_utilities


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
            PT.T: [120.0, 60.0],
            PT.H_NET_A: [1.0, 1.0],
            PT.H_NET_HOT: [2.0, 2.0],
            PT.H_NET_COLD: [3.0, 3.0],
        }
    )

    monkeypatch.setattr(
        hp,
        "create_problem_table_with_t_int",
        lambda streams, is_shifted=True: ProblemTable({PT.T: [120.0, 60.0]}),
    )
    monkeypatch.setattr(
        hp,
        "get_utility_heat_cascade",
        lambda **_kwargs: {
            "T_col": np.array([120.0, 60.0]),
            "updates": {
                PT.H_NET_UT: np.array([0.0, 0.0]),
                PT.H_HOT_UT: np.array([0.0, 0.0]),
                PT.H_COLD_UT: np.array([0.0, 0.0]),
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
        idx=0,
    )
    assert isinstance(out, ProblemTable)
    np.testing.assert_allclose(out[PT.H_NET_HOT], expected_hot)
    np.testing.assert_allclose(out[PT.H_NET_COLD], expected_cold)
    np.testing.assert_allclose(out[PT.H_NET_W_AIR], expected_w_air)


def test_calc_hpr_cascade_uses_shared_temperature_intervals_for_hpr_and_air(
    monkeypatch,
):
    pt = ProblemTable(
        {
            PT.T: [120.0, 60.0],
            PT.H_NET_A: [1.0, 1.0],
            PT.H_NET_HOT: [2.0, 2.0],
            PT.H_NET_COLD: [3.0, 3.0],
        }
    )

    monkeypatch.setattr(
        hp,
        "create_problem_table_with_t_int",
        lambda streams, is_shifted=True: ProblemTable({PT.T: [120.0, 100.0, 60.0]}),
    )
    monkeypatch.setattr(
        hp,
        "get_process_heat_cascade",
        lambda **_kwargs: ProblemTable(
            {
                PT.T: [120.0, 90.0, 60.0],
                PT.H_NET: [6.0, 3.0, 0.0],
                PT.H_NET_HOT: [0.0, 3.0, 6.0],
                PT.H_NET_COLD: [6.0, 3.0, 0.0],
            }
        ),
    )
    monkeypatch.setattr(
        hp,
        "get_utility_heat_cascade",
        lambda **kwargs: {
            "T_col": np.array(kwargs["T_int_vals"], dtype=float),
            "updates": {
                PT.H_NET_UT: np.zeros(len(kwargs["T_int_vals"])),
                PT.H_HOT_UT: np.zeros(len(kwargs["T_int_vals"])),
                PT.H_COLD_UT: np.zeros(len(kwargs["T_int_vals"])),
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

    np.testing.assert_allclose(out[PT.T], np.array([120.0, 100.0, 90.0, 60.0]))
    np.testing.assert_allclose(out[PT.H_NET_W_AIR], np.array([7.0, 5.0, 4.0, 1.0]))
    np.testing.assert_allclose(out[PT.H_NET_HOT], np.array([2.0, 4.0, 5.0, 8.0]))
    np.testing.assert_allclose(out[PT.H_NET_COLD], np.array([9.0, 7.0, 6.0, 3.0]))


def test_calc_hpr_cascade_forwards_selected_idx_to_nested_helpers(monkeypatch):
    pt = ProblemTable(
        {
            PT.T: [120.0, 60.0],
            PT.H_NET_A: [1.0, 1.0],
            PT.H_NET_HOT: [2.0, 2.0],
            PT.H_NET_COLD: [3.0, 3.0],
        }
    )
    calls = {}

    monkeypatch.setattr(
        hp,
        "create_problem_table_with_t_int",
        lambda **kwargs: (
            calls.__setitem__("grid_idx", kwargs.get("idx"))
            or ProblemTable({PT.T: [120.0, 60.0]})
        ),
    )
    monkeypatch.setattr(
        hp,
        "get_process_heat_cascade",
        lambda **kwargs: (
            calls.__setitem__("air_idx", kwargs.get("idx"))
            or ProblemTable(
                {
                    PT.T: [120.0, 60.0],
                    PT.H_NET: [0.0, 0.0],
                    PT.H_NET_HOT: [0.0, 0.0],
                    PT.H_NET_COLD: [0.0, 0.0],
                }
            )
        ),
    )
    monkeypatch.setattr(
        hp,
        "get_utility_heat_cascade",
        lambda **kwargs: (
            calls.__setitem__("utility_idx", kwargs.get("idx"))
            or {
                "T_col": np.array(kwargs["T_int_vals"], dtype=float),
                "updates": {
                    PT.H_NET_UT: np.zeros(len(kwargs["T_int_vals"])),
                    PT.H_HOT_UT: np.zeros(len(kwargs["T_int_vals"])),
                    PT.H_COLD_UT: np.zeros(len(kwargs["T_int_vals"])),
                },
            }
        ),
    )

    amb_streams = StreamCollection()
    amb_streams.add(Stream(name="Air", t_supply=50.0, t_target=80.0, heat_flow=10.0))
    res = SimpleNamespace(
        hpr_hot_streams=StreamCollection(),
        hpr_cold_streams=StreamCollection(),
        amb_streams=amb_streams,
    )

    hp._calc_hpr_cascade(
        pt,
        res,
        is_T_vals_shifted=True,
        is_heat_pumping=True,
        idx=3,
    )

    assert calls == {"grid_idx": 3, "air_idx": 3, "utility_idx": 3}


def test_get_hpr_targets_forwards_selected_idx_to_preprocessing(monkeypatch):
    captured = {}
    _patch_output_model_validate(monkeypatch)

    monkeypatch.setattr(
        hp,
        "construct_HPRTargetInputs",
        lambda *args, idx=0, **kwargs: (
            captured.__setitem__("idx", idx) or _base_args(idx=idx)
        ),
    )
    monkeypatch.setitem(
        hp._HP_PLACEMENT_HANDLERS,
        HPRcycle.MultiTempCarnot.value,
        lambda args: SimpleNamespace(to_output_payload=lambda: {"idx": args.idx}),
    )

    out = hp._get_hpr_targets(
        Q_hpr_target=50.0,
        T_vals=np.array([120.0, 80.0]),
        H_hot=np.array([0.0, -10.0]),
        H_cold=np.array([10.0, 0.0]),
        zone_config=SimpleNamespace(HPR_TYPE=HPRcycle.MultiTempCarnot.value),
        is_heat_pumping=True,
        idx=2,
    )

    assert captured["idx"] == 2
    assert out["idx"] == 2


def test_compute_indirect_hpr_uses_idx_not_state_id_for_utility_profile(monkeypatch):
    zone = Zone(name="Plant", type=ZT.S.value, zone_config=Configuration())
    zone.set_state_context({"0": 0, "peak": 1}, [1.0, 1.0], 2)
    zone.targets[TT.TS.value] = SimpleNamespace(pt=ProblemTable({PT.T: [120.0, 60.0]}))
    calls = {}

    monkeypatch.setattr(hp, "_validate_hpr_required", lambda *args, **kwargs: 25.0)
    monkeypatch.setattr(
        hp,
        "get_process_heat_cascade",
        lambda **kwargs: (
            calls.__setitem__("profile_kwargs", kwargs)
            or ProblemTable(
                {
                    PT.T: [120.0, 60.0],
                    PT.H_NET_HOT: [0.0, -10.0],
                    PT.H_NET_COLD: [10.0, 0.0],
                }
            )
        ),
    )
    monkeypatch.setattr(
        hp,
        "_get_hpr_targets",
        lambda **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(hp, "_calc_hpr_cascade", lambda **kwargs: kwargs["pt"])
    monkeypatch.setattr(hp, "_get_hpr_graphs", lambda **kwargs: {})
    monkeypatch.setattr(
        hp,
        "_get_hpr_target_summary",
        lambda res, target_zone: {
            "hpr_cycle": "stub",
            "hpr_utility_total": 11.0,
            "hpr_work": 2.0,
            "hpr_external_utility": 3.0,
            "hpr_ambient_hot": 4.0,
            "hpr_ambient_cold": 5.0,
            "hpr_cop": 6.0,
            "hpr_eta_he": 7.0,
            "hpr_success": True,
            "hpr_hot_streams": StreamCollection(),
            "hpr_cold_streams": StreamCollection(),
            "hpr_details": {},
        },
    )
    monkeypatch.setattr(
        hp,
        "_get_hpr_residual_utility_summary",
        lambda **kwargs: {
            "hot_utilities": StreamCollection(),
            "cold_utilities": StreamCollection(),
            "hot_utility_target": 0.0,
            "cold_utility_target": 0.0,
            "heat_recovery_target": 0.0,
            "heat_recovery_limit": None,
            "degree_of_int": None,
            "utility_cost": 0.0,
            "hot_pinch": None,
            "cold_pinch": None,
        },
    )
    monkeypatch.setattr(
        hp.IndirectHeatPumpTarget,
        "model_validate",
        classmethod(lambda cls, value: value),
    )

    payload = hp.compute_indirect_heat_pump_or_refrigeration_target(
        zone,
        is_heat_pumping=True,
        args={"state_id": "peak", "idx": 1},
    )

    assert calls["profile_kwargs"]["idx"] == 1
    assert "state_id" not in calls["profile_kwargs"]
    assert payload["state_id"] == "peak"
    assert payload["state_idx"] == 1


def test_indirect_hpr_load_uses_finite_utility_profile_when_base_target_has_nans(
    monkeypatch,
):
    zone = Zone(name="Plant", type=ZT.S.value, zone_config=Configuration())
    zone.targets[TT.TS.value] = SimpleNamespace(
        pt=ProblemTable(
            {
                PT.T: [120.0, 60.0],
                PT.H_NET_HOT: [np.nan, np.nan],
                PT.H_NET_COLD: [np.nan, np.nan],
            }
        )
    )
    utility_profile = ProblemTable(
        {
            PT.T: [120.0, 60.0],
            PT.H_NET_HOT: [0.0, -40.0],
            PT.H_NET_COLD: [100.0, 0.0],
        }
    )
    captured = {}

    monkeypatch.setattr(
        hp,
        "get_process_heat_cascade",
        lambda **_kwargs: utility_profile,
    )
    monkeypatch.setattr(
        hp,
        "_get_hpr_targets",
        lambda **kwargs: captured.__setitem__("target_load", kwargs["Q_hpr_target"])
        or SimpleNamespace(),
    )
    monkeypatch.setattr(hp, "_calc_hpr_cascade", lambda **kwargs: kwargs["pt"])
    monkeypatch.setattr(hp, "_get_hpr_graphs", lambda **kwargs: {})
    monkeypatch.setattr(
        hp,
        "_get_hpr_target_summary",
        lambda res, target_zone: {
            "hpr_cycle": "stub",
            "hpr_utility_total": 11.0,
            "hpr_work": 2.0,
            "hpr_external_utility": 3.0,
            "hpr_ambient_hot": 4.0,
            "hpr_ambient_cold": 5.0,
            "hpr_cop": 6.0,
            "hpr_eta_he": 7.0,
            "hpr_success": True,
            "hpr_hot_streams": StreamCollection(),
            "hpr_cold_streams": StreamCollection(),
            "hpr_details": {},
        },
    )
    monkeypatch.setattr(
        hp,
        "_get_hpr_residual_utility_summary",
        lambda **kwargs: {
            "hot_utilities": StreamCollection(),
            "cold_utilities": StreamCollection(),
            "hot_utility_target": 0.0,
            "cold_utility_target": 0.0,
            "heat_recovery_target": 0.0,
            "heat_recovery_limit": None,
            "degree_of_int": None,
            "utility_cost": 0.0,
            "hot_pinch": None,
            "cold_pinch": None,
        },
    )
    monkeypatch.setattr(
        hp.IndirectHeatPumpTarget,
        "model_validate",
        classmethod(lambda cls, value: value),
    )

    payload = hp.compute_indirect_heat_pump_or_refrigeration_target(
        zone,
        is_heat_pumping=True,
    )

    assert payload["hpr_success"] is True
    assert captured["target_load"] == pytest.approx(100.0)


def test_validate_hpr_required_ignores_nan_load_entries():
    pt = ProblemTable(
        {
            PT.T: [120.0, 80.0, 60.0],
            PT.H_NET_HOT: [np.nan, -20.0, -40.0],
            PT.H_NET_COLD: [np.nan, 100.0, 0.0],
        }
    )
    config = Configuration(options={"HPR_LOAD_VALUE": 0.25})

    assert hp._validate_hpr_required(
        H_net_cold=pt[PT.H_NET_COLD],
        H_net_hot=pt[PT.H_NET_HOT],
        is_heat_pumping=True,
        zone_config=config,
    ) == pytest.approx(25.0)


def test_validate_hpr_required_returns_zero_for_all_nan_load_entries():
    pt = ProblemTable(
        {
            PT.T: [120.0, 80.0],
            PT.H_NET_HOT: [np.nan, np.nan],
            PT.H_NET_COLD: [np.nan, np.nan],
        }
    )

    assert (
        hp._validate_hpr_required(
            H_net_cold=pt[PT.H_NET_COLD],
            H_net_hot=pt[PT.H_NET_HOT],
            is_heat_pumping=True,
            zone_config=Configuration(),
        )
        == 0.0
    )


def test_hpr_residual_utility_summary_retargets_direct_utilities():
    hot_utilities, cold_utilities = _make_base_utility_collections()
    base_target = SimpleNamespace(
        hot_utilities=hot_utilities,
        cold_utilities=cold_utilities,
        hot_utility_target=40.0,
        cold_utility_target=30.0,
        heat_recovery_target=10.0,
        heat_recovery_limit=50.0,
    )
    pt = ProblemTable(
        {
            PT.T: [120.0, 80.0, 60.0, 20.0],
            PT.H_NET_W_AIR: [40.0, 0.0, 0.0, 30.0],
            PT.H_NET_HP: [15.0, 0.0, 0.0, 10.0],
        }
    )

    summary = hp._get_hpr_residual_utility_summary(
        pt=pt,
        base_target=base_target,
        idx=0,
        is_direct=True,
        is_heat_pumping=True,
    )

    assert summary["hot_utility_target"] == pytest.approx(25.0)
    assert summary["cold_utility_target"] == pytest.approx(20.0)
    assert summary["heat_recovery_target"] == pytest.approx(25.0)
    assert summary["degree_of_int"] == pytest.approx(0.5)
    assert summary["hot_pinch"] == pytest.approx(80.0)
    assert summary["cold_pinch"] == pytest.approx(60.0)
    assert float(summary["hot_utilities"][0].heat_flow[0]) == pytest.approx(25.0)
    assert float(summary["cold_utilities"][0].heat_flow[0]) == pytest.approx(20.0)
    np.testing.assert_allclose(
        pt[PT.H_NET_HOT_AFTR_HP], np.array([0.0, 0.0, 0.0, -20.0])
    )
    np.testing.assert_allclose(
        pt[PT.H_NET_COLD_AFTR_HP], np.array([25.0, 0.0, 0.0, 0.0])
    )


def test_hpr_residual_utility_summary_retargets_indirect_utilities():
    hot_utilities, cold_utilities = _make_base_utility_collections()
    base_target = SimpleNamespace(
        hot_utilities=hot_utilities,
        cold_utilities=cold_utilities,
        hot_utility_target=40.0,
        cold_utility_target=30.0,
        heat_recovery_target=10.0,
        heat_recovery_limit=50.0,
    )
    pt = ProblemTable(
        {
            PT.T: [120.0, 80.0, 60.0, 20.0],
            PT.H_NET_UT: [40.0, 0.0, 0.0, 30.0],
            PT.H_NET_HP: [15.0, 0.0, 0.0, 10.0],
            PT.RCP_UT_NET: [0.0, 0.0, 0.0, 0.0],
        }
    )

    summary = hp._get_hpr_residual_utility_summary(
        pt=pt,
        base_target=base_target,
        idx=0,
        is_direct=False,
        is_heat_pumping=True,
    )

    assert summary["hot_utility_target"] == pytest.approx(25.0)
    assert summary["cold_utility_target"] == pytest.approx(20.0)
    assert summary["heat_recovery_target"] == pytest.approx(25.0)
    assert summary["degree_of_int"] == pytest.approx(0.5)
    assert summary["hot_pinch"] == pytest.approx(80.0)
    assert summary["cold_pinch"] == pytest.approx(60.0)
    assert float(summary["hot_utilities"][0].heat_flow[0]) == pytest.approx(25.0)
    assert float(summary["cold_utilities"][0].heat_flow[0]) == pytest.approx(20.0)
    np.testing.assert_allclose(
        pt[PT.H_NET_HOT_UT_AFTR_HP], np.array([25.0, 0.0, 0.0, 0.0])
    )
    np.testing.assert_allclose(
        pt[PT.H_NET_COLD_UT_AFTR_HP], np.array([0.0, 0.0, 0.0, -20.0])
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
        idx=0,
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
            PT.T: [120.0, 80.0],
            PT.H_NET_HOT: [8.0, 2.0],
            PT.H_NET_COLD: [1.0, 6.0],
            PT.H_HOT_UT: [0.0, 3.0],
            PT.H_COLD_UT: [4.0, 0.0],
            PT.H_HOT_HP: [0.0, 5.0],
            PT.H_COLD_HP: [3.0, 0.0],
            PT.H_NET_W_AIR: [4.0, 1.0],
            PT.H_NET_HP: [2.0, -2.0],
        }
    )

    graphs = hp._get_hpr_graphs(pt, is_direct=True, is_heat_pumping=True)

    assert set(graphs) == {GT.NLP_HP.value, GT.GCC_HP.value}
    assert list(graphs[GT.NLP_HP.value].columns) == [
        PT.T.value,
        PT.H_NET_HOT.value,
        PT.H_NET_COLD.value,
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
    nlp_hp_graph = next(
        graph for graph in graph_set["graphs"] if graph["type"] == GT.NLP_HP.value
    )
    hpr_segment_titles = {segment["title"] for segment in nlp_hp_graph["segments"]}
    assert "Heat Pump Condenser" in hpr_segment_titles
    assert "Heat Pump Evaporator" in hpr_segment_titles


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
