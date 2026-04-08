"""Focused branch coverage tests for ``OpenPinch.main`` helper routines."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import OpenPinch.main as main_mod
from OpenPinch.classes.stream import Stream
from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.classes.zone import Zone
from OpenPinch.lib.enums import ZoneType


class _DummyTarget:
    def __init__(self, name: str):
        self.name = name

    def serialize_json(self):
        return {"name": self.name}


def _utility_collection():
    sc = StreamCollection()
    sc.add(
        Stream(
            name="HU",
            t_supply=180.0,
            t_target=170.0,
            heat_flow=10.0,
            is_process_stream=False,
        )
    )
    sc.add(
        Stream(
            name="CU",
            t_supply=20.0,
            t_target=30.0,
            heat_flow=9.0,
            is_process_stream=False,
        )
    )
    sc.add(
        Stream(
            name="Other",
            t_supply=40.0,
            t_target=50.0,
            heat_flow=8.0,
            is_process_stream=False,
        )
    )
    return sc


def test_get_targets_raises_on_unknown_identifier():
    zone = Zone(name="bad", identifier="UNKNOWN")
    with pytest.raises(ValueError, match="No valid zone passed"):
        main_mod.get_targets(zone)


def test_get_visualise_legacy_path(monkeypatch):
    calls = []

    def _fake_visualise_graphs(graph_set, graph):
        calls.append((graph_set["name"], graph))
        graph_set["graphs"].append(graph)

    monkeypatch.setattr(main_mod, "visualise_graphs", _fake_visualise_graphs)
    data = [SimpleNamespace(name="A", graphs=[{"id": 1}, {"id": 2}])]

    res = main_mod.get_visualise(data)
    assert res["graphs"][0]["name"] == "A"
    assert len(res["graphs"][0]["graphs"]) == 2
    assert len(calls) == 2


def test_unit_operation_targets_covers_direct_and_invalid_nesting(monkeypatch):
    calls = []
    monkeypatch.setattr(
        main_mod, "compute_direct_integration_targets", lambda z: calls.append(z.name)
    )

    valid = Zone(name="UO", identifier=ZoneType.O.value)
    valid.config.DO_DIRECT_OPERATION_TARGETING = True
    child = Zone(name="child", identifier=ZoneType.O.value)
    valid.add_zone(child)
    out = main_mod._get_unit_operation_targets(valid)
    assert out is valid
    assert set(calls) == {"UO", "child"}

    invalid = Zone(name="UO_bad", identifier=ZoneType.O.value)
    invalid.config.DO_DIRECT_OPERATION_TARGETING = True
    invalid.add_zone(Zone(name="bad", identifier="X"))
    with pytest.raises(ValueError, match="Unit operation zones can only contain"):
        main_mod._get_unit_operation_targets(invalid)


def test_process_targets_covers_invalid_and_indirect_paths(monkeypatch):
    invalid = Zone(name="P_bad", identifier=ZoneType.P.value)
    invalid.add_zone(Zone(name="bad", identifier="X"))
    with pytest.raises(ValueError, match="Process zones can only contain"):
        main_mod._get_process_targets(invalid)

    calls = []
    monkeypatch.setattr(
        main_mod,
        "compute_direct_integration_targets",
        lambda z: calls.append(f"direct:{z.name}"),
    )
    monkeypatch.setattr(
        main_mod,
        "compute_indirect_integration_targets",
        lambda z: calls.append(f"indirect:{z.name}"),
    )
    monkeypatch.setattr(
        main_mod,
        "_get_unit_operation_targets",
        lambda z: calls.append(f"uo:{z.name}") or z,
    )

    proc = Zone(name="P", identifier=ZoneType.P.value)
    proc.add_zone(Zone(name="op", identifier=ZoneType.O.value))
    proc.add_zone(Zone(name="sub_proc", identifier=ZoneType.P.value))
    proc.config.DO_INDIRECT_PROCESS_TARGETING = True
    out = main_mod._get_process_targets(proc)

    assert out is proc
    assert any("direct:P" == c for c in calls)
    assert any("indirect:P" == c for c in calls)


def test_site_targets_covers_branching_and_invalid_nesting(monkeypatch):
    monkeypatch.setattr(main_mod, "compute_direct_integration_targets", lambda z: z)
    monkeypatch.setattr(main_mod, "compute_indirect_integration_targets", lambda z: z)

    invalid = Zone(name="S_bad", identifier=ZoneType.S.value)
    invalid.add_zone(Zone(name="bad", identifier="X"))
    with pytest.raises(ValueError, match="Sites zones can only contain"):
        main_mod._get_site_targets(invalid)

    calls = []
    monkeypatch.setattr(
        main_mod,
        "compute_direct_integration_targets",
        lambda z: calls.append(f"direct:{z.name}"),
    )
    monkeypatch.setattr(
        main_mod,
        "compute_indirect_integration_targets",
        lambda z: calls.append(f"indirect:{z.name}"),
    )
    monkeypatch.setattr(
        main_mod,
        "_get_unit_operation_targets",
        lambda z: calls.append(f"uo:{z.name}") or z,
    )
    monkeypatch.setattr(
        main_mod, "_get_process_targets", lambda z: calls.append(f"proc:{z.name}") or z
    )

    site = Zone(name="S", identifier=ZoneType.S.value)
    site.add_zone(Zone(name="op", identifier=ZoneType.O.value))
    site.add_zone(Zone(name="proc", identifier=ZoneType.P.value))
    site.add_zone(Zone(name="sub_site", identifier=ZoneType.S.value))
    out = main_mod._get_site_targets(site)

    assert out is site
    assert "uo:op" in calls
    assert "proc:proc" in calls
    assert any(c.startswith("direct:") for c in calls)
    assert "indirect:S" in calls


def test_community_and_regional_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(
        main_mod, "_get_site_targets", lambda z: calls.append(f"site:{z.name}") or z
    )

    community = Zone(name="C", identifier=ZoneType.C.value)
    community.add_zone(Zone(name="S1", identifier=ZoneType.S.value))
    out_c = main_mod._get_community_targets(community)
    assert out_c is community
    assert "site:S1" in calls

    monkeypatch.setattr(
        main_mod,
        "_get_community_targets",
        lambda z: calls.append(f"community:{z.name}") or z,
    )
    regional = Zone(name="R", identifier=ZoneType.R.value)
    regional.add_zone(Zone(name="C1", identifier=ZoneType.C.value))
    out_r = main_mod._get_regional_targets(regional)
    assert out_r is regional
    assert "community:C1" in calls


def test_report_and_utility_extractors_and_extract_results(monkeypatch):
    root = Zone(name="Root", identifier=ZoneType.P.value)
    child = Zone(name="Child", identifier=ZoneType.P.value)
    root.add_zone(child)

    root._targets = {"t0": _DummyTarget("root_target")}
    child._targets = {"t1": _DummyTarget("child_target")}

    report = main_mod._get_report(root)
    assert {"name": "root_target"} in report
    assert {"name": "child_target"} in report

    hot = StreamCollection()
    cold = StreamCollection()
    for s in _utility_collection():
        if s.name.startswith("H"):
            hot.add(s)
        elif s.name.startswith("C"):
            cold.add(s)
        else:
            hot.add(s)
    root.hot_utilities = hot
    root.cold_utilities = cold

    defaults = main_mod._get_utilities(root)
    assert defaults[0] is not None and defaults[0].name == "HU"
    assert defaults[1] is not None and defaults[1].name == "CU"

    monkeypatch.setattr(main_mod, "get_output_graph_data", lambda _z: {"graph": []})
    extracted = main_mod.extract_results(root)
    assert extracted["name"] == "Root"
    assert "targets" in extracted
    assert "utilities" in extracted
    assert extracted["graphs"] == {"graph": []}
