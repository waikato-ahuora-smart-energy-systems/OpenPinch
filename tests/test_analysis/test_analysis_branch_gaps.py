"""Targeted branch-coverage tests for remaining analysis module gaps."""

from __future__ import annotations

import builtins
from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.analysis import capital_cost_and_area_targeting as cca
from OpenPinch.analysis import data_preparation as dp
from OpenPinch.analysis import direct_integration_entry as di
from OpenPinch.analysis import gcc_manipulation as gm
from OpenPinch.analysis import graph_data as gd
from OpenPinch.analysis import indirect_integration_entry as ii
from OpenPinch.analysis import problem_table_analysis as pta
from OpenPinch.analysis import temperature_driving_force as tdf
from OpenPinch.analysis import utility_targeting as ut
from OpenPinch.classes import ProblemTable, Stream, StreamCollection, Zone
from OpenPinch.lib.config import Configuration
from OpenPinch.lib.enums import (
    GraphType as GT,
    ProblemTableLabel as PT,
    StreamLoc,
    TargetType,
    ZoneType,
    ZoneType as Z,
)
from OpenPinch.lib.schema import StreamSchema, ZoneTreeSchema
from OpenPinch.utils.miscellaneous import key_name


def _stream_schema(name: str, zone: str, heat_flow: float = 100.0) -> StreamSchema:
    return StreamSchema.model_validate(
        {
            "name": name,
            "zone": zone,
            "t_supply": 200.0 if heat_flow >= 0 else 80.0,
            "t_target": 100.0 if heat_flow >= 0 else 160.0,
            "heat_flow": heat_flow,
            "dt_cont": 5.0,
            "htc": 1.0,
        }
    )


def _stream(
    name: str,
    t_supply: float,
    t_target: float,
    *,
    heat_flow: float,
    is_process_stream: bool = True,
) -> Stream:
    return Stream(
        name=name,
        t_supply=t_supply,
        t_target=t_target,
        heat_flow=heat_flow,
        dt_cont=0.0,
        htc=1.0,
        is_process_stream=is_process_stream,
    )


def test_capital_cost_targets_and_area_targeting_branches(monkeypatch):
    cfg = Configuration()
    capital_cost, annual_capital_cost = cca.get_capital_cost_targets(100.0, 3, cfg)
    assert capital_cost > 0.0
    assert annual_capital_cost > 0.0

    tdf_payload = {
        "delta_T1": np.array([20.0, 15.0]),
        "delta_T2": np.array([10.0, 10.0]),
        "dh_vals": np.array([4.0, 6.0]),
        "t_h1": np.array([190.0, 140.0]),
        "t_h2": np.array([160.0, 110.0]),
        "t_c1": np.array([180.0, 130.0]),
        "t_c2": np.array([155.0, 105.0]),
    }
    monkeypatch.setattr(
        cca, "get_temperature_driving_forces", lambda *args, **kwargs: tdf_payload
    )
    monkeypatch.setattr(
        cca,
        "_map_interval_resistances_to_tdf",
        lambda *args, **kwargs: np.array([2.0, 4.0]),
    )
    monkeypatch.setattr(cca, "compute_LMTD_from_dts", lambda dt1, dt2: np.array([12.0]))
    with pytest.raises(
        ValueError, match="Shape of heat exchanger area calculation arrays are unequal"
    ):
        cca.get_area_targets(
            np.array([200.0, 150.0, 100.0]),
            np.array([10.0, 5.0, 0.0]),
            np.array([12.0, 7.0, 2.0]),
            np.array([0.0, 1.0, 1.0]),
            np.array([0.0, 1.0, 1.0]),
        )

    monkeypatch.setattr(
        cca, "compute_LMTD_from_dts", lambda dt1, dt2: np.array([12.0, 10.0])
    )
    area = cca.get_area_targets(
        np.array([200.0, 150.0, 100.0]),
        np.array([10.0, 5.0, 0.0]),
        np.array([12.0, 7.0, 2.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
    )
    assert area > 0.0


def test_map_interval_resistances_to_tdf_maps_expected_intervals():
    mapped = cca._map_interval_resistances_to_tdf(
        T_vals=np.array([200.0, 150.0, 100.0]),
        R_hot_bal=np.array([0.0, 2.0, 4.0]),
        R_cold_bal=np.array([0.0, 1.0, 3.0]),
        t_h1=np.array([190.0, 140.0]),
        t_h2=np.array([160.0, 110.0]),
        t_c1=np.array([180.0, 130.0]),
        t_c2=np.array([155.0, 105.0]),
    )
    np.testing.assert_allclose(mapped, np.array([3.0, 7.0]))


def test_data_preparation_zone_rewrite_and_config_branches(monkeypatch):
    zone_tree = ZoneTreeSchema(name="Root", type="", children=None)
    _, zone_type = dp._get_validated_zone_info(zone_tree, depth=2)
    assert zone_type == ZoneType.O.value

    # cover children=None root handling, empty zone labels, empty components and collision naming loop
    stream_collision = _stream_schema("Root", "Root")
    stream_canonical = _stream_schema("Canon", "Root/Root_2")
    stream_empty_zone = SimpleNamespace(name="Empty", zone="")
    stream_slash_zone = SimpleNamespace(name="Slash", zone=" / ")
    dp._rewrite_stream_zones_from_tree(
        zone_tree,
        [stream_collision, stream_canonical, stream_empty_zone, stream_slash_zone],
    )
    assert zone_tree.children is not None
    assert stream_collision.zone == "Root/Root_2"
    assert stream_canonical.zone == "Root/Root_2"

    # cover recursive path collection and unique suffix path resolution
    suffix_tree = ZoneTreeSchema(
        name="Root",
        type=ZoneType.S.value,
        children=[
            ZoneTreeSchema(
                name="Area",
                type=ZoneType.P.value,
                children=[
                    ZoneTreeSchema(name="Unit", type=ZoneType.P.value, children=None)
                ],
            )
        ],
    )
    suffix_stream = _stream_schema("S1", "Area/Unit")
    dp._rewrite_stream_zones_from_tree(suffix_tree, [suffix_stream])
    assert suffix_stream.zone == "Root/Area/Unit"

    # cover _build_full_path empty path branch and O-name collision handling
    stream_root_slash = _stream_schema("R1", "/")
    stream_conflict_1 = _stream_schema("C1", "A/O1")
    stream_conflict_2 = _stream_schema("C2", "A")
    monkeypatch.setattr(
        dp,
        "sorted",
        lambda iterable, key=None: builtins.sorted(iterable, key=key, reverse=True),
        raising=False,
    )
    tree = dp._validate_zone_tree_structure(
        None,
        [stream_root_slash, stream_conflict_1, stream_conflict_2],
        top_zone_name="Site",
    )
    assert tree.name == "Site"
    assert stream_root_slash.zone.startswith("Site/")
    assert stream_conflict_2.zone.endswith("/A/O2")

    cfg = Configuration()
    cfg.DO_TURBINE_WORK = True
    cfg.P_TURBINE_BOX = 300
    cfg.DT_PHASE_CHANGE = 0.0
    cfg.DT_CONT = -1.0
    cfg = dp._validate_config_data_completed(cfg)
    assert cfg.P_TURBINE_BOX == 200
    assert cfg.DT_PHASE_CHANGE == pytest.approx(0.01)
    assert cfg.DT_CONT == 0.0


def test_data_preparation_process_stream_assignment_branches():
    cfg = Configuration()
    root = Zone(name="Site", identifier=ZoneType.S.value, zone_config=cfg)
    area = Zone(
        name="Area", identifier=ZoneType.P.value, zone_config=cfg, parent_zone=root
    )
    unit = Zone(
        name="Unit", identifier=ZoneType.P.value, zone_config=cfg, parent_zone=area
    )
    root.add_zone(area, sub=True)
    area.add_zone(unit, sub=True)

    no_zone_stream = SimpleNamespace(zone=None)
    rel_zone_stream = _stream_schema("HotRel", "Area/Unit", heat_flow=50.0)
    out = dp._get_process_streams_in_each_subzone(
        root, [no_zone_stream, rel_zone_stream]
    )
    assert out.subzones["Area"].subzones["Unit"].hot_streams[0].name == "HotRel"


def test_problem_table_analysis_uncovered_branches(monkeypatch):
    hot = StreamCollection()
    cold = StreamCollection()
    hot.add(_stream("H", 200.0, 120.0, heat_flow=100.0))
    cold.add(_stream("C", 80.0, 140.0, heat_flow=-80.0))

    captured = {}
    pt = ProblemTable(
        {
            PT.T.value: [200.0, 100.0],
            PT.H_HOT.value: [10.0, 0.0],
            PT.H_COLD.value: [10.0, 0.0],
            PT.H_NET.value: [0.0, 0.0],
        }
    )

    def _fake_create(streams, is_shifted=True, zone_config=None):
        captured["streams"] = streams
        return pt.copy

    orig_create_problem_table_with_t_int = pta.create_problem_table_with_t_int
    monkeypatch.setattr(pta, "create_problem_table_with_t_int", _fake_create)
    monkeypatch.setattr(pta, "problem_table_algorithm", lambda **kwargs: kwargs["pt"])
    monkeypatch.setattr(pta, "get_heat_recovery_target_from_pt", lambda _pt: 0.0)
    out = pta.get_process_heat_cascade(
        hot_streams=hot, cold_streams=cold, all_streams=None
    )
    assert isinstance(out, ProblemTable)
    assert isinstance(captured["streams"], StreamCollection)
    assert len(captured["streams"]) == len(hot + cold)

    cfg = Configuration()
    cfg.DO_PROCESS_HP_TARGETING = True
    cfg.T_ENV = 25.0
    cfg.DT_ENV_CONT = 5.0
    cfg.DT_PHASE_CHANGE = 2.0
    pt_hp = orig_create_problem_table_with_t_int(
        streams=[_stream("H2", 150.0, 100.0, heat_flow=30.0)],
        is_shifted=False,
        zone_config=cfg,
    )
    assert 20.0 in pt_hp.col[PT.T.value]
    assert 18.0 in pt_hp.col[PT.T.value]
    assert 30.0 in pt_hp.col[PT.T.value]
    assert 32.0 in pt_hp.col[PT.T.value]

    tiny_pt = ProblemTable({PT.T.value: [100.0], PT.H_HOT.value: [10.0]})
    assert pta._get_T_start_on_opposite_cc(tiny_pt, 5.0, PT.H_HOT.value) is None

    no_cross_pt = ProblemTable(
        {PT.T.value: [120.0, 100.0, 80.0], PT.H_HOT.value: [10.0, 9.0, 8.0]}
    )
    assert pta._get_T_start_on_opposite_cc(no_cross_pt, 100.0, PT.H_HOT.value) is None


def test_gcc_manipulation_uncovered_branches(monkeypatch):
    pt = ProblemTable(
        {
            PT.T.value: [150.0, 100.0],
            PT.H_NET.value: [5.0, 0.0],
            PT.H_COLD.value: [6.0, 0.0],
            PT.H_HOT.value: [1.0, 0.0],
            PT.H_NET_NP.value: [5.0, 0.0],
            PT.H_NET_A.value: [5.0, 0.0],
        }
    )

    orig_get_ggc_pockets = gm.get_GGC_pockets
    monkeypatch.setattr(gm, "get_GCC_without_pockets", lambda x: x)
    monkeypatch.setattr(
        gm,
        "get_GCC_with_vertical_heat_transfer",
        lambda *args, **kwargs: {PT.H_NET_V.value: np.array([4.0, 0.0])},
    )
    monkeypatch.setattr(
        gm, "get_GGC_pockets", lambda _pt: {PT.H_NET_PK.value: np.array([1.0, 0.0])}
    )
    monkeypatch.setattr(
        gm, "get_GCC_needing_utility", lambda h: {PT.H_NET_A.value: np.asarray(h)}
    )
    monkeypatch.setattr(
        gm,
        "get_seperated_gcc_heat_load_profiles",
        lambda h, **kwargs: {
            PT.H_NET_HOT.value: np.asarray(h),
            PT.H_NET_COLD.value: np.asarray(h),
        },
    )
    out = gm.get_additional_GCCs(pt, do_vert_cc_calc=True, do_assisted_ht_calc=True)
    assert PT.H_NET_V.value in out.columns
    assert PT.H_NET_PK.value in out.columns

    same = gm.get_GCC_with_partial_pockets(pt)
    assert same is pt

    pocket_pt = ProblemTable(
        {PT.H_NET.value: [8.0, 5.0], PT.H_NET_NP.value: [3.0, 2.0]}
    )
    pockets = orig_get_ggc_pockets(pocket_pt)
    np.testing.assert_allclose(pockets[PT.H_NET_PK.value], np.array([5.0, 3.0]))
    assert PT.H_NET_PK.value in pocket_pt.columns

    assert gm._pocket_exit_index(np.array([10.0, 9.0, 8.0]), 2, 0, -1) == 0


def test_graph_data_uncovered_branches():
    graph_set = {"graphs": []}
    graph = SimpleNamespace(
        type=GT.BCC.value,
        data={
            PT.T.value: [120.0, 80.0],
            PT.H_HOT_BAL.value: [5.0, 0.0],
            PT.H_COLD_BAL.value: [5.0, 0.0],
        },
    )
    gd.visualise_graphs(graph_set, graph)
    assert graph_set["graphs"][0]["type"] == GT.BCC.value

    assert gd._normalise_gcc_fields("x") == ["x"]
    assert gd._normalise_gcc_flags(None, 2) == [False, False]
    with pytest.raises(ValueError, match="must have the same length"):
        gd._normalise_gcc_flags([True], 2)

    with pytest.raises(
        ValueError, match="Unrecognised composite curve stream location"
    ):
        gd._graph_cc(GT.CC.value, StreamLoc.Unassigned, [120.0, 80.0], [0.0, 5.0])

    assert gd._column_to_list({PT.T.value: [1.0, 2.0]}, PT.T) == [1.0, 2.0]

    class _ColLike:
        columns = [PT.T.value]
        col = {PT.T.value: [3.0, 4.0]}

    assert gd._column_to_list(_ColLike(), PT.T.value) == [3.0, 4.0]

    class _IndexOnly:
        def __getitem__(self, key):
            if key == "x":
                return (7.0, 8.0)
            raise KeyError(key)

    assert gd._column_to_list(_IndexOnly(), "x") == [7.0, 8.0]
    with pytest.raises(KeyError, match="Column 'missing' not found"):
        gd._column_to_list({}, "missing")

    assert (
        gd._classify_segment(float("nan"), is_utility_profile=False)
        == StreamLoc.Unassigned
    )
    assert gd._segment_streamloc(StreamLoc.ColdS.value) == StreamLoc.ColdS
    assert gd._segment_streamloc(StreamLoc.HotS.value) == StreamLoc.HotS
    assert gd._segment_streamloc(StreamLoc.HotU.value) == StreamLoc.HotU
    assert gd._segment_streamloc(StreamLoc.ColdU.value) == StreamLoc.ColdU
    assert gd._segment_streamloc("unknown") == StreamLoc.Unassigned


def test_temperature_driving_force_uncovered_branches():
    with pytest.raises(ValueError, match="must be the same length"):
        tdf.get_temperature_driving_forces(
            np.array([100.0, 90.0]),
            np.array([0.0]),
            np.array([80.0]),
            np.array([0.0]),
        )

    with pytest.raises(ValueError, match="cannot be empty"):
        tdf.get_temperature_driving_forces(
            np.array([]),
            np.array([]),
            np.array([80.0]),
            np.array([0.0]),
        )

    with pytest.raises(ValueError, match="must be the same length"):
        tdf._normalise_curve(np.array([0.0, 1.0]), np.array([100.0]))

    assert tdf._discontinuity_values(np.array([1.0])).size == 0


def test_utility_targeting_uncovered_branches():
    assert (
        ut._target_utility([], np.array([100.0, 90.0]), np.array([0.0, 0.0]), 0, 1)
        == []
    )
    assert (
        ut._maximise_utility_duty(
            np.array([100.0]),
            np.array([0.0]),
            Ts=120.0,
            Tt=80.0,
            is_hot_ut=True,
            Q_assigned=0.0,
        )
        == 0.0
    )


def test_direct_integration_helper_and_entry_branches(monkeypatch):
    hu = StreamCollection()
    hu.add(_stream("HU", 150.0, 130.0, heat_flow=1.0, is_process_stream=False))
    cu = StreamCollection()
    cu.add(_stream("CU", 60.0, 90.0, heat_flow=1.0, is_process_stream=False))
    with pytest.raises(ValueError, match="Infeasible temperature interval"):
        di._create_net_hot_and_cold_stream_collections_for_site_analysis(
            np.array([100.0, 100.0]),
            np.array([0.0, 10.0]),
            hu,
            cu,
        )

    net = StreamCollection()
    assert di._add_net_segment_stateful(100.0, 80.0, -1, 10.0, [], [], net, 1) == (
        -1,
        1,
    )

    one_u = [_stream("U", 120.0, 100.0, heat_flow=10.0, is_process_stream=False)]
    rem0 = [0.0]
    idx, k = di._add_net_segment_stateful(
        120.0, 100.0, 0, 5.0, one_u, rem0, StreamCollection(), 1
    )
    assert (idx, k) == (0, 1)

    two_u = [
        _stream("U1", 120.0, 100.0, heat_flow=0.0, is_process_stream=False),
        _stream("U2", 120.0, 100.0, heat_flow=8.0, is_process_stream=False),
    ]
    idx, k = di._add_net_segment_stateful(
        120.0,
        100.0,
        0,
        3.0,
        two_u,
        [0.0, 8.0],
        StreamCollection(),
        1,
    )
    assert (idx, k) == (1, 2)

    rem1 = [5.0]
    idx, k = di._add_net_segment_stateful(
        120.0, 100.0, 0, 10.0, one_u, rem1, StreamCollection(), 1
    )
    assert (idx, k) == (0, 2)
    assert di._find_next_available_utility(0, [], []) == -1

    cfg = Configuration()
    cfg.DO_BALANCED_CC = True
    cfg.DO_AREA_TARGETING = True
    cfg.DO_PROCESS_HP_TARGETING = True
    cfg.HP_LOAD_FRACTION = 0.8

    zone = Zone(name="Plant", identifier=Z.P.value, zone_config=cfg)
    zone.hot_streams.add(_stream("H", 200.0, 120.0, heat_flow=100.0))
    zone.cold_streams.add(_stream("C", 80.0, 160.0, heat_flow=-80.0))
    zone.hot_utilities.add(
        _stream("HU1", 250.0, 200.0, heat_flow=10.0, is_process_stream=False)
    )
    zone.cold_utilities.add(
        _stream("CU1", 20.0, 40.0, heat_flow=10.0, is_process_stream=False)
    )
    zone.add_target_from_results = lambda *_args, **_kwargs: None

    pt_stub = ProblemTable(
        {
            PT.T.value: [200.0, 100.0],
            PT.DELTA_T.value: [0.0, 100.0],
            PT.H_HOT.value: [50.0, 0.0],
            PT.H_COLD.value: [50.0, 0.0],
            PT.H_NET.value: [10.0, 0.0],
            PT.H_NET_HOT.value: [0.0, -20.0],
            PT.H_NET_COLD.value: [20.0, 0.0],
            PT.H_NET_A.value: [10.0, 0.0],
            PT.H_HOT_UT.value: [10.0, 0.0],
            PT.H_COLD_UT.value: [10.0, 0.0],
            PT.RCP_HOT.value: [0.0, 1.0],
            PT.RCP_COLD.value: [0.0, 1.0],
            PT.RCP_HOT_UT.value: [0.0, 1.0],
            PT.RCP_COLD_UT.value: [0.0, 1.0],
        }
    )
    monkeypatch.setattr(di, "config", cfg)
    monkeypatch.setattr(di, "get_process_heat_cascade", lambda **kwargs: pt_stub.copy)
    monkeypatch.setattr(di, "get_heat_recovery_target_from_pt", lambda _pt: 0.0)
    monkeypatch.setattr(
        di, "set_zonal_targets", lambda **kwargs: {"hot_utility_target": 1.0}
    )
    monkeypatch.setattr(di, "get_additional_GCCs", lambda pt, **kwargs: pt)
    monkeypatch.setattr(
        di, "_validate_heat_pump_targeting_required", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        di,
        "get_heat_pump_targets",
        lambda **kwargs: {
            "hp_hot_streams": StreamCollection(),
            "hp_cold_streams": StreamCollection(),
            "Q_amb": 0.0,
        },
    )
    monkeypatch.setattr(di, "calc_heat_pump_cascade", lambda **kwargs: kwargs["pt"])
    monkeypatch.setattr(di, "get_utility_targets", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        di,
        "_create_net_hot_and_cold_stream_collections_for_site_analysis",
        lambda *args, **kwargs: (StreamCollection(), StreamCollection()),
    )
    monkeypatch.setattr(
        di,
        "get_balanced_CC",
        lambda *args, **kwargs: {
            PT.H_HOT_BAL.value: np.array([10.0, 0.0]),
            PT.H_COLD_BAL.value: np.array([10.0, 0.0]),
            PT.R_HOT_BAL.value: np.array([0.0, 1.0]),
            PT.R_COLD_BAL.value: np.array([0.0, 1.0]),
        },
    )
    monkeypatch.setattr(di, "get_min_number_hx", lambda *args, **kwargs: 3)
    monkeypatch.setattr(di, "get_area_targets", lambda *args, **kwargs: 120.0)
    monkeypatch.setattr(
        di, "get_capital_cost_targets", lambda *args, **kwargs: (1000.0, 100.0)
    )
    monkeypatch.setattr(di, "_save_graph_data", lambda *args, **kwargs: {})

    out = di.compute_direct_integration_targets(zone)
    assert out is zone


def test_indirect_integration_heat_pump_branch_and_sum_branch(monkeypatch):
    cfg = Configuration()
    cfg.DO_UTILITY_HP_TARGETING = True
    cfg.HP_LOAD_FRACTION = 0.8

    zone = Zone(name="Site", identifier=Z.S.value, zone_config=cfg)
    zone.net_hot_streams = StreamCollection()
    zone.net_cold_streams = StreamCollection()
    zone.add_target_from_results = lambda *_args, **_kwargs: None
    zone.import_hot_and_cold_streams_from_sub_zones = lambda **kwargs: None

    tz_key = key_name(zone.name, TargetType.TZ.value)
    tz_target = SimpleNamespace(
        hot_utilities=StreamCollection(),
        cold_utilities=StreamCollection(),
        heat_recovery_target=30.0,
        heat_recovery_limit=40.0,
        hot_utility_target=20.0,
    )
    tz_target.hot_utilities.add(
        _stream("HU", 250.0, 200.0, heat_flow=5.0, is_process_stream=False)
    )
    tz_target.cold_utilities.add(
        _stream("CU", 30.0, 70.0, heat_flow=5.0, is_process_stream=False)
    )
    zone.targets[tz_key] = tz_target

    pt_stub = ProblemTable(
        {
            PT.T.value: [200.0, 100.0],
            PT.H_HOT.value: [50.0, 0.0],
            PT.H_COLD.value: [50.0, 0.0],
            PT.H_NET_UT.value: [10.0, 5.0],
            PT.H_HOT_UT.value: [8.0, 0.0],
            PT.H_COLD_UT.value: [4.0, 0.0],
            PT.H_NET_HOT.value: [0.0, 0.0],
            PT.H_NET_COLD.value: [0.0, 0.0],
            PT.H_NET_HP_UT.value: [0.0, 0.0],
        }
    )

    orig_sum_subzone_targets = ii._sum_subzone_targets
    monkeypatch.setattr(ii, "_sum_subzone_targets", lambda _zone: _zone)
    monkeypatch.setattr(ii, "get_process_heat_cascade", lambda **kwargs: pt_stub.copy)
    monkeypatch.setattr(ii, "get_heat_recovery_target_from_pt", lambda _pt: 0.0)
    monkeypatch.setattr(
        ii, "set_zonal_targets", lambda **kwargs: {"hot_utility_target": 1.0}
    )
    monkeypatch.setattr(
        ii,
        "_get_site_process_heat_load_profiles",
        lambda *args, **kwargs: {
            PT.H_NET_HOT.value: np.array([0.0, 0.0]),
            PT.H_NET_COLD.value: np.array([0.0, 0.0]),
        },
    )
    monkeypatch.setattr(
        ii,
        "_get_site_utility_heat_cascade",
        lambda *args, **kwargs: {
            PT.H_NET_UT.value: np.array([10.0, 5.0]),
            PT.H_HOT_UT.value: np.array([8.0, 0.0]),
            PT.H_COLD_UT.value: np.array([4.0, 0.0]),
        },
    )
    monkeypatch.setattr(
        ii, "_match_utility_gen_and_use_at_same_level", lambda hu, cu: (hu, cu)
    )
    monkeypatch.setattr(
        ii, "_validate_heat_pump_targeting_required", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        ii,
        "get_heat_pump_targets",
        lambda **kwargs: {
            "hp_hot_streams": StreamCollection(),
            "hp_cold_streams": StreamCollection(),
            "Q_amb": 0.0,
        },
    )
    monkeypatch.setattr(ii, "calc_heat_pump_cascade", lambda **kwargs: kwargs["pt"])
    monkeypatch.setattr(ii, "_save_graph_data", lambda *args, **kwargs: {})
    monkeypatch.setattr(ii, "_compute_utility_cost", lambda *args, **kwargs: 0.0)

    out = ii.compute_indirect_integration_targets(zone)
    assert out is zone

    # cover _sum_subzone_targets area-accumulation branch by forcing tol below zero
    parent = Zone(
        name="Parent", identifier=ZoneType.S.value, zone_config=Configuration()
    )
    parent.add_target_from_results = lambda *_args, **_kwargs: None
    parent.hot_utilities.add(
        _stream("HU0", 250.0, 200.0, heat_flow=0.0, is_process_stream=False)
    )
    parent.cold_utilities.add(
        _stream("CU0", 20.0, 40.0, heat_flow=0.0, is_process_stream=False)
    )
    parent.targets[key_name(parent.name, TargetType.DI.value)] = SimpleNamespace(
        heat_recovery_limit=10.0
    )

    child = Zone(
        name="Child",
        identifier=ZoneType.P.value,
        zone_config=parent.config,
        parent_zone=parent,
    )
    child_target = SimpleNamespace(
        hot_utility_target=10.0,
        cold_utility_target=5.0,
        heat_recovery_target=4.0,
        utility_cost=3.0,
        hot_utilities=[
            _stream("HUc", 240.0, 210.0, heat_flow=2.0, is_process_stream=False)
        ],
        cold_utilities=[
            _stream("CUc", 30.0, 60.0, heat_flow=1.0, is_process_stream=False)
        ],
        num_units=7,
        area=12.0,
    )
    child.targets[key_name(child.name, TargetType.DI.value)] = child_target
    parent.add_zone(child, sub=True)

    monkeypatch.setattr(ii, "tol", -1e-9)
    orig_sum_subzone_targets(parent)
