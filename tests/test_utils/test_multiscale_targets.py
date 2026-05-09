"""Focused branch coverage tests for ``OpenPinch.multiscale_targets`` routines."""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from OpenPinch import *
from OpenPinch.utils import multiscale_targeting as ms
from OpenPinch.lib import *
from OpenPinch.classes.stream import Stream
from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.classes.zone import Zone
from OpenPinch.lib.enums import ZT


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


def test_unit_operation_targets_covers_direct_and_invalid_nesting(monkeypatch):
    calls = []
    direct = lambda z: calls.append(z.name)

    valid = Zone(name="UO", type=ZT.O.value)
    valid.config.DO_OPERATION_LEVEL_TARGETING = True
    child = Zone(name="child", type=ZT.O.value)
    valid.add_zone(child)
    out = ms._get_unit_operation_targets(valid, direct_service_func=direct)
    assert out is valid
    assert calls.count("UO") == 2

    invalid = Zone(name="UO_bad", type=ZT.O.value)
    invalid.config.DO_OPERATION_LEVEL_TARGETING = True
    invalid.add_zone(Zone(name="bad", type="X"))
    with pytest.raises(ValueError, match="Unit operation zones can only contain"):
        ms._get_unit_operation_targets(invalid)


def test_process_targets_covers_invalid_and_indirect_paths(monkeypatch):
    invalid = Zone(name="P_bad", type=ZT.P.value)
    invalid.add_zone(Zone(name="bad", type="X"))
    with pytest.raises(ValueError, match="Process zones can only contain"):
        ms._get_process_targets(invalid)

    calls = []
    direct = lambda z: calls.append(f"direct:{z.name}")
    indirect = lambda z: calls.append(f"indirect:{z.name}")
    monkeypatch.setattr(
        ms,
        "_get_unit_operation_targets",
        lambda z, direct_service_func=None, indirect_service_func=None: (
            calls.append(f"uo:{z.name}") or z
        ),
    )
    monkeypatch.setattr(
        ms,
        "_get_process_targets",
        ms._get_process_targets,
    )

    proc = Zone(name="P", type=ZT.P.value)
    proc.add_zone(Zone(name="op", type=ZT.O.value))
    proc.add_zone(Zone(name="sub_proc", type=ZT.P.value))
    proc.config.DO_INDIRECT_PROCESS_TARGETING = True
    out = ms._get_process_targets(
        proc,
        direct_service_func=direct,
        indirect_service_func=indirect,
    )

    assert out is proc
    assert any("direct:P" == c for c in calls)
    assert any("indirect:P" == c for c in calls)


def test_site_targets_covers_branching_and_invalid_nesting(monkeypatch):
    invalid = Zone(name="S_bad", type=ZT.S.value)
    invalid.add_zone(Zone(name="bad", type="X"))
    with pytest.raises(ValueError, match="Sites zones can only contain"):
        ms._get_site_targets(invalid)

    calls = []
    direct = lambda z: calls.append(f"direct:{z.name}")
    indirect = lambda z: calls.append(f"indirect:{z.name}")
    monkeypatch.setattr(
        ms,
        "_get_unit_operation_targets",
        lambda z, direct_service_func=None, indirect_service_func=None: (
            calls.append(f"uo:{z.name}") or z
        ),
    )
    monkeypatch.setattr(
        ms,
        "_get_process_targets",
        lambda z, direct_service_func=None, indirect_service_func=None: (
            calls.append(f"proc:{z.name}") or z
        ),
    )
    monkeypatch.setattr(
        ms,
        "_get_site_targets",
        ms._get_site_targets,
    )

    site = Zone(name="S", type=ZT.S.value)
    site.add_zone(Zone(name="op", type=ZT.O.value))
    site.add_zone(Zone(name="proc", type=ZT.P.value))
    site.add_zone(Zone(name="sub_site", type=ZT.S.value))
    out = ms._get_site_targets(
        site,
        direct_service_func=direct,
        indirect_service_func=indirect,
    )

    assert out is site
    assert "uo:op" in calls
    assert "proc:proc" in calls
    assert any(c.startswith("direct:") for c in calls)
    assert "indirect:S" in calls


def test_community_and_regional_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ms,
        "_get_site_targets",
        lambda z, direct_service_func=None, indirect_service_func=None: (
            calls.append(f"site:{z.name}") or z
        ),
    )

    community = Zone(name="C", type=ZT.C.value)
    community.add_zone(Zone(name="S1", type=ZT.S.value))
    out_c = ms._get_community_targets(
        community,
        direct_service_func=lambda z: z,
        indirect_service_func=lambda z: z,
    )
    assert out_c is community
    assert "site:S1" in calls

    monkeypatch.setattr(
        ms,
        "_get_community_targets",
        lambda z, direct_service_func=None, indirect_service_func=None: (
            calls.append(f"community:{z.name}") or z
        ),
    )
    regional = Zone(name="R", type=ZT.R.value)
    regional.add_zone(Zone(name="C1", type=ZT.C.value))
    out_r = ms._get_regional_targets(
        regional,
        direct_service_func=lambda z: z,
        indirect_service_func=lambda z: z,
    )
    assert out_r is regional
    assert "community:C1" in calls


def test_report_and_utility_extractors_and_extract_results(monkeypatch):
    root = Zone(name="Root", type=ZT.P.value)
    child = Zone(name="Child", type=ZT.P.value)
    root.add_zone(child)

    root._targets = {"t0": _DummyTarget("root_target")}
    child._targets = {"t1": _DummyTarget("child_target")}

    report = ms._get_report(root)
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

    defaults = ms._get_utilities(root)
    assert defaults[0] is not None and defaults[0].name == "HU"
    assert defaults[1] is not None and defaults[1].name == "CU"

    monkeypatch.setattr(ms, "get_output_graph_data", lambda _z: {"graph": []})
    extracted = ms.extract_results(root)
    assert extracted["name"] == "Root"
    assert "targets" in extracted
    assert "utilities" in extracted
    assert extracted["graphs"] == {"graph": []}
