"""Regression tests for the pinch problem classes."""

import json

# import types
import sys
from pathlib import Path

import pytest

from OpenPinch.classes.accessors.plot_accessor import _PlotAccessor
from OpenPinch.classes.pinch_problem import PinchProblem
from OpenPinch.classes.zone import Zone
from OpenPinch.resources import copy_sample_case


@pytest.fixture
def sample_problem():
    """Return sample problem data used by this test module."""
    return {
        "options": {"DT_CONT": 10},
        "streams": [
            {
                "zone": "Z1",
                "name": "H1",
                "t_supply": 150,
                "t_target": 60,
                "heat_flow": 2.0,
                "dtcont": 1,
            }
        ],
        "utilities": [{"name": "LP Steam", "t_supply": 150, "type": "Hot", "cost": 20}],
    }


def test_load_json(tmp_path: Path, sample_problem):
    p = tmp_path / "problem.json"
    p.write_text(json.dumps(sample_problem), encoding="utf-8")

    obj = PinchProblem()
    out = obj.load(p)

    assert isinstance(out, Zone)
    assert obj.problem_filepath == p
    assert obj.results is None
    # to_problem_json should mirror problem_data
    assert obj.to_problem_json() == sample_problem


def test_load_excel_calls_reader_and_sets_path(
    monkeypatch, tmp_path: Path, sample_problem
):
    # Patch the module-level symbol imported in the class' module:
    # inside PinchProblem, get_problem_from_excel is a *name* in the module namespace
    mod = sys.modules[PinchProblem.__module__]
    called = {}

    def fake_excel_reader(path, output_json):
        called["args"] = (path, output_json)
        return sample_problem

    monkeypatch.setattr(mod, "get_problem_from_excel", fake_excel_reader, raising=True)

    x = tmp_path / "problem.xlsx"
    x.write_bytes(b"")

    obj = PinchProblem()
    out = obj.load(x)

    assert isinstance(out, Zone)
    assert obj.problem_filepath == x
    assert called["args"][0] == x
    assert called["args"][1] is None


def test_load_csv_tuple_calls_reader(monkeypatch, tmp_path: Path, sample_problem):
    mod = sys.modules[PinchProblem.__module__]
    called = {}

    def fake_csv_reader(streams_csv, utilities_csv, output_json):
        called["args"] = (streams_csv, utilities_csv, output_json)
        return sample_problem

    monkeypatch.setattr(mod, "get_problem_from_csv", fake_csv_reader, raising=True)

    s = tmp_path / "streams.csv"
    s.write_text("name,t_supply,t_target,cp\n", encoding="utf-8")
    u = tmp_path / "utilities.csv"
    u.write_text("name,T,cost\n", encoding="utf-8")

    obj = PinchProblem()
    out = obj.load((s, u))

    assert isinstance(out, Zone)
    # CSV tuple is "not a single-file source"
    assert obj.problem_filepath is None
    assert (
        called["args"][0] == s and called["args"][1] == u and called["args"][2] is None
    )


def test_load_csv_directory_bundle(monkeypatch, tmp_path: Path, sample_problem):
    mod = sys.modules[PinchProblem.__module__]
    called = {}

    def fake_csv_reader(streams_csv, utilities_csv, output_json):
        called["args"] = (streams_csv, utilities_csv, output_json)
        return sample_problem

    monkeypatch.setattr(mod, "get_problem_from_csv", fake_csv_reader, raising=True)

    d = tmp_path / "bundle"
    d.mkdir()
    (d / "streams.csv").write_text("name,t_supply,t_target,cp\n", encoding="utf-8")
    (d / "utilities.csv").write_text("name,T,cost\n", encoding="utf-8")

    obj = PinchProblem()
    out = obj.load(d)

    assert isinstance(out, Zone)
    assert obj.problem_filepath == d
    s_csv, u_csv, out_json = called["args"]
    assert s_csv == d / "streams.csv"
    assert u_csv == d / "utilities.csv"
    assert out_json is None


def test_load_csv_directory_missing_files_raises(tmp_path: Path):
    d = tmp_path / "bundle"
    d.mkdir()
    # intentionally missing utilities.csv
    (d / "streams.csv").write_text("", encoding="utf-8")

    obj = PinchProblem()
    with pytest.raises(FileNotFoundError):
        obj.load(d)


def test_load_unrecognized_source_raises(tmp_path: Path):
    weird = tmp_path / "problem.txt"
    weird.write_text("not valid", encoding="utf-8")

    obj = PinchProblem()
    with pytest.raises(ValueError):
        obj.load(weird)


def test_target_raises_without_problem_loaded():
    obj = PinchProblem()
    with pytest.raises(RuntimeError):
        obj.target()


def test_export_without_results_dir_raises(sample_problem, capsys):
    obj = PinchProblem()
    # Simulate loaded data so export triggers the "no results_dir" error path
    obj._problem_data = sample_problem
    obj._results = {"ok": True}
    with pytest.raises(ValueError):
        obj.export_excel(None)

    captured = capsys.readouterr()
    assert captured.out == ""


def test_export_calls_writer_with_results(monkeypatch, tmp_path: Path):
    mod = sys.modules[PinchProblem.__module__]
    called = {}

    def fake_writer(*, target_response, master_zone, out_dir):
        called["kwargs"] = {
            "target_response": target_response,
            "master_zone": master_zone,
            "out_dir": out_dir,
        }
        output_path = Path(out_dir) / "targets.xlsx"
        # we don't actually write, just return a path
        return output_path

    monkeypatch.setattr(
        mod, "export_target_summary_to_excel_with_units", fake_writer, raising=True
    )

    obj = PinchProblem()
    obj._results = {"foo": "bar"}  # Pretend targeting already ran
    dest = tmp_path / "out"

    path = obj.export_excel(dest)

    assert path == dest / "targets.xlsx"
    assert called["kwargs"]["target_response"] == {"foo": "bar"}
    assert called["kwargs"]["master_zone"] is None
    assert called["kwargs"]["out_dir"] == dest
    # object state updated
    assert obj.results_dir == dest


def test_to_problem_json_without_data_raises():
    obj = PinchProblem()
    with pytest.raises(RuntimeError):
        obj.to_problem_json()


def test_repr_changes_with_state(tmp_path: Path):
    obj = PinchProblem()
    r = repr(obj)
    assert "<in-memory or CSV tuple>" in r
    assert "export=<unset>" in r
    assert "results=no" in r

    # set paths / results and check
    obj.results_dir = tmp_path / "out"
    obj._results = {}
    r2 = repr(obj)
    assert str(obj.results_dir) in r2
    assert "results=yes" in r2


def test_from_json_builds_and_roundtrips(sample_problem):
    obj = PinchProblem.from_json(sample_problem)
    assert obj.to_problem_json() == sample_problem


def test_load_in_memory_mapping_builds_zone(sample_problem):
    obj = PinchProblem()

    out = obj.load(sample_problem)

    assert isinstance(out, Zone)
    assert obj.problem_data == sample_problem


def test_load_json_normalises_missing_zone_and_name(tmp_path: Path):
    payload = {
        "options": {},
        "utilities": [
            {"name": None, "type": "Hot"},
            {"name": "", "type": "Cold"},
            {"name": "HP Steam", "type": "Hot"},
        ],
        "streams": [
            {"zone": "Zone A", "name": "H1", "t_supply": 150},
            {"zone": None, "name": None, "t_supply": 140},
            {"zone": "", "name": "", "t_supply": 130},
        ],
    }
    p = tmp_path / "problem.json"
    p.write_text(json.dumps(payload), encoding="utf-8")

    obj = PinchProblem()
    with pytest.raises(ValueError):
        obj.load(p)


def test_init_with_problem_filepath_calls_load(monkeypatch, tmp_path: Path):
    called = {"source": None}

    def fake_load(self, source):
        called["source"] = source
        return {}

    monkeypatch.setattr(PinchProblem, "load", fake_load)
    p = tmp_path / "p.json"
    p.write_text("{}", encoding="utf-8")
    PinchProblem().load(source=p)

    assert called["source"] == p


def test_load_json_parse_error_raises_value_error(tmp_path: Path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not valid json", encoding="utf-8")
    obj = PinchProblem()
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        obj.load(broken)


def test_export_calls_target_when_results_missing(
    monkeypatch, tmp_path: Path, sample_problem
):
    mod = sys.modules[PinchProblem.__module__]
    called = {"target": 0, "write": 0}

    def fake_target(self):
        called["target"] += 1
        self._results = {"ok": 1}
        self._master_zone = {"zone": 1}
        return self._results

    def fake_writer(*, target_response, master_zone, out_dir):
        called["write"] += 1
        return Path(out_dir) / "export.xlsx"

    monkeypatch.setattr(PinchProblem, "target", fake_target)
    monkeypatch.setattr(
        mod, "export_target_summary_to_excel_with_units", fake_writer, raising=True
    )

    obj = PinchProblem()
    obj._problem_data = sample_problem
    out = obj.export_excel(tmp_path)

    assert called["target"] == 1
    assert called["write"] == 1
    assert out == tmp_path / "export.xlsx"


def test_problem_data_and_master_zone_properties():
    obj = PinchProblem()
    obj._problem_data = {"a": 1}
    obj._master_zone = {"z": 1}
    assert obj.problem_data == {"a": 1}
    assert obj.master_zone == {"z": 1}


def test_root_stream_views_are_exposed_on_problem():
    payload = {
        "options": {"DT_CONT": 10},
        "streams": [
            {
                "zone": "Area1",
                "name": "H1",
                "t_supply": 180.0,
                "t_target": 90.0,
                "heat_flow": 100.0,
                "dtcont": 5.0,
            },
            {
                "zone": "Area1",
                "name": "C1",
                "t_supply": 40.0,
                "t_target": 120.0,
                "heat_flow": 80.0,
                "dtcont": 5.0,
            },
        ],
        "utilities": [
            {"name": "Steam", "t_supply": 220.0, "type": "Hot", "cost": 20.0},
            {"name": "CW", "t_supply": 25.0, "type": "Cold", "cost": 5.0},
        ],
    }

    problem = PinchProblem(source=payload, project_name="Site")

    assert problem.hot_streams is problem.master_zone.hot_streams
    assert problem.cold_streams is problem.master_zone.cold_streams
    assert problem.hot_utilities is problem.master_zone.hot_utilities
    assert problem.cold_utilities is problem.master_zone.cold_utilities


def test_root_stream_views_require_loaded_problem():
    problem = PinchProblem()

    with pytest.raises(RuntimeError, match="No input loaded"):
        _ = problem.hot_streams


def test_problem_hot_stream_temperature_mutation_updates_root_zone_stream():
    payload = {
        "options": {"DT_CONT": 10},
        "streams": [
            {
                "zone": "Area1",
                "name": "H1",
                "t_supply": 180.0,
                "t_target": 90.0,
                "heat_flow": 100.0,
                "dtcont": 5.0,
            }
        ],
        "utilities": [],
    }

    problem = PinchProblem(source=payload, project_name="Site")

    hot_stream = problem.hot_streams["Area1.H1"]
    hot_stream.t_supply = 195.0

    assert problem.hot_streams["Area1.H1"].t_supply == pytest.approx(195.0)
    assert problem.master_zone.hot_streams["Area1.H1"].t_supply == pytest.approx(195.0)
    assert hot_stream.t_max == pytest.approx(195.0)


def test_show_dashboard_builds_graph_payload(monkeypatch):
    mod = sys.modules[PinchProblem.__module__]
    captured = {}

    class GraphWithDump:
        def model_dump(self):
            return {"x": 1}

    class ResultContainer:
        graphs = {"g1": GraphWithDump(), "g2": {"y": 2}}

    def fake_render(zone, graph_payload, page_title, value_rounding):
        captured["zone"] = zone
        captured["payload"] = graph_payload
        captured["title"] = page_title
        captured["rounding"] = value_rounding

    monkeypatch.setattr(mod, "_render_streamlit_dashboard", fake_render, raising=True)
    obj = PinchProblem()
    obj._master_zone = {"name": "root"}
    obj._results = ResultContainer()

    obj.show_dashboard(page_title="Dash", value_rounding=3)

    assert captured["zone"] == {"name": "root"}
    assert captured["payload"]["g1"] == {"x": 1}
    assert captured["payload"]["g2"] == {"y": 2}
    assert captured["title"] == "Dash"
    assert captured["rounding"] == 3


def test_show_dashboard_requires_available_zone():
    obj = PinchProblem()
    with pytest.raises(RuntimeError, match="No analysed zone is available"):
        obj.show_dashboard()


def test_init_accepts_problem_filepath_and_run(monkeypatch, tmp_path: Path):
    called = {"load": None}

    def fake_load(self, source=None):
        called["load"] = source
        return {"zone": "ok"}

    monkeypatch.setattr(PinchProblem, "load", fake_load)

    case_path = tmp_path / "case.json"
    case_path.write_text("{}", encoding="utf-8")

    PinchProblem(source=case_path)

    assert called["load"] == case_path


def test_target_accessor_supports_named_workflow(monkeypatch):
    called = {}

    def fake_execute_targeting(
        self,
        *,
        target_id,
        application_zone=None,
        options=None,
        include_subzones=False,
        single_zone_service=None,
        direct_service_func=None,
        indirect_service_func=None,
    ):
        called["target_id"] = target_id
        called["application_zone"] = application_zone
        called["options"] = options
        called["include_subzones"] = include_subzones
        called["single_zone_service"] = single_zone_service
        called["direct_service_func"] = direct_service_func
        called["indirect_service_func"] = indirect_service_func
        return {"target": "di"}

    monkeypatch.setattr(
        PinchProblem,
        "_execute_targeting",
        fake_execute_targeting,
    )

    obj = PinchProblem()
    out = obj.target.direct_heat_integration(
        zone_name="Plant/DI",
        options={"dt_min": 15},
    )

    assert out == {"target": "di"}
    assert called["target_id"] == "Direct Integration"
    assert called["application_zone"] == "Plant/DI"
    assert called["options"] == {"dt_min": 15}
    assert called["include_subzones"] is False


def test_target_accessor_include_subzones_uses_run_targeting(monkeypatch):
    called = {}

    def fake_execute_targeting(
        self,
        *,
        target_id,
        application_zone=None,
        options=None,
        include_subzones=False,
        single_zone_service=None,
        direct_service_func=None,
        indirect_service_func=None,
    ):
        called["target_id"] = target_id
        called["application_zone"] = application_zone
        called["options"] = options
        called["include_subzones"] = include_subzones
        called["single_zone_service"] = single_zone_service
        called["direct_service_func"] = direct_service_func
        called["indirect_service_func"] = indirect_service_func
        return {"target": "di"}

    monkeypatch.setattr(PinchProblem, "_execute_targeting", fake_execute_targeting)

    obj = PinchProblem()
    out = obj.target.direct_heat_integration(
        zone_name="Plant/DI",
        include_subzones=True,
    )

    assert out == {"target": "di"}
    assert called["target_id"] == "Direct Integration"
    assert called["application_zone"] == "Plant/DI"
    assert called["include_subzones"] is True
    assert called["direct_service_func"] is not None
    assert called["indirect_service_func"] is None


def test_validate_uses_schema_and_prepare_problem(monkeypatch, sample_problem):
    mod = sys.modules[PinchProblem.__module__]
    calls = {}

    class DummyPayload:
        streams = ["s"]
        utilities = ["u"]
        options = {"x": 1}
        zone_tree = {"name": "z"}

    monkeypatch.setattr(
        mod.TargetInput,
        "model_validate",
        classmethod(lambda cls, value: DummyPayload()),
    )
    monkeypatch.setattr(
        mod, "_validate_problem_semantics", lambda payload, context: None
    )

    def fake_prepare_problem(**kwargs):
        calls["kwargs"] = kwargs
        return {"zone": "ok"}

    monkeypatch.setattr(
        sys.modules["OpenPinch.services.input_data_processing.data_preparation"],
        "prepare_problem",
        fake_prepare_problem,
        raising=True,
    )

    obj = PinchProblem()
    obj._problem_data = sample_problem
    obj._project_name = "Example"

    payload = obj.validate()

    assert payload.streams == ["s"]


def test_summary_frame_compact_and_detailed(monkeypatch):
    class _Value:
        def __init__(self, value, units="kW"):
            self.value = value
            self.units = units

    class _Utility:
        def __init__(self, name, heat_flow):
            self.name = name
            self.heat_flow = heat_flow

    target = type(
        "Target",
        (),
        {
            "name": "Plant/DI",
            "Qh": _Value(10.0),
            "Qc": _Value(20.0),
            "Qr": _Value(30.0),
            "temp_pinch": type(
                "TempPinch",
                (),
                {"hot_temp": _Value(110.0, "degC"), "cold_temp": _Value(90.0, "degC")},
            )(),
            "hot_utilities": [_Utility("Steam", _Value(10.0))],
            "cold_utilities": [_Utility("CW", _Value(20.0))],
        },
    )()
    results = type("Results", (), {"targets": [target]})()

    monkeypatch.setattr(PinchProblem, "target", lambda self: results)
    monkeypatch.setattr(
        sys.modules[PinchProblem.__module__],
        "build_summary_dataframe",
        lambda targets: __import__("pandas").DataFrame([{"Target": targets[0].name}]),
        raising=True,
    )

    obj = PinchProblem()
    compact = obj.summary_frame()
    detailed = obj.summary_frame(detailed=True)

    assert list(compact.columns[:4]) == [
        "Target",
        "Hot Utility Target",
        "Cold Utility Target",
        "Heat Recovery",
    ]
    assert compact.iloc[0]["Hot Utilities"] == "Steam: 10.00"
    assert detailed.iloc[0]["Target"] == "Plant/DI"


def test_summary_frame_preserves_equal_hot_and_cold_pinch_values():
    class _Value:
        def __init__(self, value, units="kW"):
            self.value = value
            self.units = units

    target = type(
        "Target",
        (),
        {
            "name": "Plant/DI",
            "Qh": _Value(10.0),
            "Qc": _Value(20.0),
            "Qr": _Value(30.0),
            "temp_pinch": type(
                "TempPinch",
                (),
                {
                    "hot_temp": _Value(120.0, "degC"),
                    "cold_temp": _Value(120.0, "degC"),
                },
            )(),
            "hot_utilities": [],
            "cold_utilities": [],
        },
    )()
    obj = PinchProblem()
    results = type("Results", (), {"targets": [target]})()
    obj.target = lambda: results

    summary = obj.summary_frame()

    assert summary.iloc[0]["Hot Pinch"] == 120.0
    assert summary.iloc[0]["Cold Pinch"] == 120.0


def test_graph_data_uses_results_then_master_zone(monkeypatch):
    obj = PinchProblem()
    obj._results = type(
        "Results",
        (),
        {
            "graphs": {
                "Plant": type(
                    "GraphSet",
                    (),
                    {
                        "model_dump": lambda self: {
                            "name": "Plant/Direct Integration",
                            "zone_name": "Plant",
                            "zone_address": "Site/Plant",
                            "graphs": [],
                        }
                    },
                )()
            }
        },
    )()
    monkeypatch.setattr(PinchProblem, "target", lambda self: obj._results)
    assert obj.plot.get_graph_data() == {
        "Plant": {
            "name": "Plant/Direct Integration",
            "zone_name": "Plant",
            "zone_address": "Site/Plant",
            "graphs": [],
        }
    }

    obj._results = type("Results", (), {"graphs": None})()
    obj._master_zone = {"zone": "ok"}
    monkeypatch.setattr(
        sys.modules[_PlotAccessor.__module__],
        "get_output_graph_data",
        lambda zone: {"Fallback": {"graphs": [{"type": "Grand Composite Curve"}]}},
        raising=True,
    )
    assert (
        obj.plot.get_graph_data()["Fallback"]["graphs"][0]["type"]
        == "Grand Composite Curve"
    )


def test_graph_catalog_and_plot_helpers(monkeypatch):
    payload = {
        "Plant/DI": {
            "name": "Plant/Direct Integration",
            "zone_name": "Plant",
            "zone_address": "Site/Plant",
            "target_type": "Direct Integration",
            "graphs": [
                {"type": "Composite Curves", "name": "Composite"},
                {"type": "Grand Composite Curve", "name": "GCC"},
            ],
        }
    }
    monkeypatch.setattr(_PlotAccessor, "get_graph_data", lambda self: payload)
    monkeypatch.setattr(
        sys.modules[_PlotAccessor.__module__],
        "_build_plotly_graph",
        lambda graph: {"built": graph["name"]},
        raising=True,
    )

    obj = PinchProblem()
    catalog = obj.plot()
    fig = obj.plot.grand_composite_curve(zone_name="Plant/DI")

    assert set(catalog["Graph Name"]) == {"Composite", "GCC"}
    assert set(catalog["Zone Address"]) == {"Site/Plant"}
    assert fig == {"built": "GCC"}


def test_plot_helper_executes_display_hook_when_available(monkeypatch):
    payload = {
        "Plant/DI": {
            "name": "Plant/Direct Integration",
            "zone_name": "Plant",
            "zone_address": "Site/Plant",
            "graphs": [
                {"type": "Composite Curves", "name": "Composite"},
            ],
        }
    }
    shown = {"count": 0}

    class FakeFigure:
        def show(self):
            shown["count"] += 1

    monkeypatch.setattr(_PlotAccessor, "get_graph_data", lambda self: payload)
    monkeypatch.setattr(
        sys.modules[_PlotAccessor.__module__],
        "_build_plotly_graph",
        lambda graph: FakeFigure(),
        raising=True,
    )

    obj = PinchProblem()
    fig = obj.plot.composite_curve(zone_name="Plant/DI", show=True)

    assert isinstance(fig, FakeFigure)
    assert shown["count"] == 1


def test_plot_helper_can_return_selected_graph_data(monkeypatch):
    payload = {
        "Plant/DI": {
            "name": "Plant/Direct Integration",
            "zone_name": "Plant",
            "zone_address": "Site/Plant",
            "graphs": [
                {"type": "Composite Curves", "name": "Composite"},
                {"type": "Grand Composite Curve", "name": "GCC"},
            ],
        }
    }
    build_calls = {"count": 0}

    def fake_build(graph):
        build_calls["count"] += 1
        return {"built": graph["name"]}

    monkeypatch.setattr(_PlotAccessor, "get_graph_data", lambda self: payload)
    monkeypatch.setattr(
        sys.modules[_PlotAccessor.__module__],
        "_build_plotly_graph",
        fake_build,
        raising=True,
    )

    obj = PinchProblem()
    graph = obj.plot.grand_composite_curve(
        zone_name="Plant/DI",
        return_graph_data=True,
    )

    assert graph == {"type": "Grand Composite Curve", "name": "GCC"}
    assert build_calls["count"] == 0


def test_plot_helper_rejects_show_when_returning_graph_data(monkeypatch):
    payload = {
        "Plant/DI": {
            "name": "Plant/Direct Integration",
            "zone_name": "Plant",
            "zone_address": "Site/Plant",
            "graphs": [
                {"type": "Composite Curves", "name": "Composite"},
            ],
        }
    }
    monkeypatch.setattr(_PlotAccessor, "get_graph_data", lambda self: payload)

    obj = PinchProblem()

    with pytest.raises(ValueError, match="show=True"):
        obj.plot.composite_curve(
            zone_name="Plant/DI",
            show=True,
            return_graph_data=True,
        )


def test_plot_helper_uses_default_notebook_dimensions(monkeypatch):
    payload = {
        "Plant/DI": {
            "name": "Plant/Direct Integration",
            "zone_name": "Plant",
            "zone_address": "Site/Plant",
            "graphs": [
                {"type": "Composite Curves", "name": "Composite", "segments": []},
            ],
        }
    }
    monkeypatch.setattr(_PlotAccessor, "get_graph_data", lambda self: payload)

    obj = PinchProblem()
    fig = obj.plot.composite_curve(zone_name="Plant/DI")

    assert fig.layout.width == 720
    assert fig.layout.height == 540
    assert fig.layout.autosize is False


def test_plot_helpers_accept_qualified_target_name_with_identifier_key(monkeypatch):
    payload = {
        "Direct Integration": {
            "name": "Plant/Direct Integration",
            "zone_name": "Plant",
            "zone_address": "Site/Plant",
            "target_type": "Direct Integration",
            "graphs": [
                {"type": "Grand Composite Curve", "name": "GCC"},
            ],
        }
    }
    monkeypatch.setattr(_PlotAccessor, "get_graph_data", lambda self: payload)
    monkeypatch.setattr(
        sys.modules[_PlotAccessor.__module__],
        "_build_plotly_graph",
        lambda graph: {"built": graph["name"]},
        raising=True,
    )

    obj = PinchProblem()
    fig = obj.plot.grand_composite_curve(zone_name="Plant/Direct Integration")

    assert fig == {"built": "GCC"}


def test_plot_helpers_match_zone_address_when_target_types_repeat(monkeypatch):
    payload = {
        "Site/AreaA/Direct Integration": {
            "name": "Site/AreaA/Direct Integration",
            "zone_name": "AreaA",
            "zone_address": "Site/AreaA",
            "target_type": "Direct Integration",
            "graphs": [
                {"type": "Grand Composite Curve", "name": "AreaA GCC"},
            ],
        },
        "Site/AreaB/Direct Integration": {
            "name": "Site/AreaB/Direct Integration",
            "zone_name": "AreaB",
            "zone_address": "Site/AreaB",
            "target_type": "Direct Integration",
            "graphs": [
                {"type": "Grand Composite Curve", "name": "AreaB GCC"},
            ],
        },
    }
    monkeypatch.setattr(_PlotAccessor, "get_graph_data", lambda self: payload)
    monkeypatch.setattr(
        sys.modules[_PlotAccessor.__module__],
        "_build_plotly_graph",
        lambda graph: {"built": graph["name"]},
        raising=True,
    )

    obj = PinchProblem()

    assert obj.plot.grand_composite_curve(zone_name="Site/AreaA") == {
        "built": "AreaA GCC"
    }
    assert obj.plot.grand_composite_curve(zone_name="Site/AreaB") == {
        "built": "AreaB GCC"
    }


def test_export_graphs_writes_html(monkeypatch, tmp_path: Path):
    payload = {
        "Plant/DI": {
            "name": "Plant/Direct Integration",
            "zone_name": "Plant",
            "zone_address": "Site/Plant",
            "graphs": [
                {"type": "Grand Composite Curve", "name": "GCC"},
            ],
        }
    }
    monkeypatch.setattr(_PlotAccessor, "get_graph_data", lambda self: payload)

    class FakeFigure:
        def write_html(self, path):
            Path(path).write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr(
        sys.modules[_PlotAccessor.__module__],
        "_build_plotly_graph",
        lambda graph: FakeFigure(),
        raising=True,
    )

    obj = PinchProblem()
    written = obj.plot.export(tmp_path)

    assert len(written) == 1
    assert written[0].exists()


def test_compare_to_builds_delta_table(monkeypatch):
    base_frame = __import__("pandas").DataFrame(
        [
            {
                "Target": "Plant/Direct Integration",
                "Hot Utility Target": 750.0,
                "Cold Utility Target": 1000.0,
                "Heat Recovery": 5150.0,
                "Hot Pinch": None,
                "Cold Pinch": 145.0,
            }
        ]
    )
    other_frame = __import__("pandas").DataFrame(
        [
            {
                "Target": "Plant/Direct Integration",
                "Hot Utility Target": 500.0,
                "Cold Utility Target": 850.0,
                "Heat Recovery": 5800.0,
                "Hot Pinch": None,
                "Cold Pinch": 170.0,
            }
        ]
    )
    base_problem = PinchProblem()
    other_problem = PinchProblem()

    monkeypatch.setattr(
        base_problem, "summary_frame", lambda detailed=False: base_frame
    )
    monkeypatch.setattr(
        other_problem,
        "summary_frame",
        lambda detailed=False: other_frame,
    )

    comparison = base_problem.compare_to(other_problem)

    assert comparison.loc["Base case", "Hot Utility Target"] == 750.0
    assert comparison.loc["Scenario", "Cold Utility Target"] == 850.0
    assert comparison.loc["Change", "Heat Recovery"] == 650.0


def test_validate_formats_schema_errors_with_stream_context(tmp_path: Path):
    payload = {
        "streams": [
            {
                "zone": "Zone A",
                "name": "Hot Feed",
                "t_supply": 150.0,
                "heat_flow": 100.0,
                "dt_cont": 10.0,
                "htc": 1.0,
            }
        ],
        "utilities": [],
        "options": {},
    }
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        PinchProblem().load(source=path)

    message = str(exc_info.value)
    assert "Input validation failed" in message
    assert "Stream 1 'Hot Feed' (entry 1)" in message
    assert "t_target" in message


def test_validate_rejects_invalid_utility_temperature_direction(tmp_path: Path):
    payload = {
        "streams": [
            {
                "zone": "Zone A",
                "name": "H1",
                "t_supply": 150.0,
                "t_target": 60.0,
                "heat_flow": 100.0,
                "dt_cont": 10.0,
                "htc": 1.0,
            }
        ],
        "utilities": [
            {
                "name": "Cooling Water",
                "type": "Cold",
                "t_supply": 35.0,
                "t_target": 25.0,
                "heat_flow": 50.0,
                "dt_cont": 5.0,
                "htc": 0.8,
                "price": 10.0,
            }
        ],
        "options": {},
    }
    path = tmp_path / "invalid_utility.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.warns() as exc_info:
        PinchProblem().load(source=path)

    message = str(exc_info[0].message)
    assert "Input validation reported 1 warning(s)" in message
    assert "Cooling Water" in message
    assert "0 issue(s)" not in message


def test_chocolate_factory_sample_can_be_copied_and_validated(tmp_path: Path):
    case_path = copy_sample_case(
        "chocolate_factory.json",
        tmp_path / "chocolate_factory.json",
    )
    problem = PinchProblem()
    problem.load(source=case_path)

    validated = problem.validate()

    assert validated.zone_tree is None
    assert len(validated.streams) == 65
    assert len(validated.utilities) == 5


def test_packaged_sample_case_name_can_be_loaded_directly():
    problem = PinchProblem(
        source="crude_preheat_train.json",
        project_name="crude_preheat_train",
    )

    validated = problem.validate()
    canonical = problem.to_problem_json(canonical=True)

    assert problem.problem_filepath == Path("crude_preheat_train.json")
    assert validated.zone_tree is None
    assert canonical["zone_tree"] is not None
    assert len(validated.streams) > 0


def test_set_dt_cont_multiplier_rebuilds_targets_and_stream_state(tmp_path: Path):
    case_path = copy_sample_case(
        "crude_preheat_train.json",
        tmp_path / "crude_preheat_train.json",
    )
    problem = PinchProblem(source=case_path, project_name=case_path.stem)

    baseline = problem.summary_frame()
    baseline_row = baseline.loc[
        baseline["Target"] == f"{case_path.stem}/Direct Integration"
    ].iloc[0]
    unit = problem.master_zone.get_subzone("Crude Unit")
    hot_stream = next(iter(unit.hot_streams))

    assert hot_stream.dt_cont == hot_stream.dt_cont_act

    problem.set_dt_cont_multiplier(0.5)

    assert problem.results is None
    assert problem.to_problem_json()["zone_tree"][
        "dt_cont_multiplier"
    ] == pytest.approx(0.5)

    unit = problem.master_zone.get_subzone("Crude Unit")
    updated_stream = next(iter(unit.hot_streams))
    assert updated_stream.dt_cont == hot_stream.dt_cont
    assert updated_stream.dt_cont_act == pytest.approx(hot_stream.dt_cont * 0.5)

    updated = problem.summary_frame()
    updated_row = updated.loc[
        updated["Target"] == f"{case_path.stem}/Direct Integration"
    ].iloc[0]

    assert updated_row["Hot Utility Target"] != baseline_row["Hot Utility Target"]
    assert updated_row["Cold Utility Target"] != baseline_row["Cold Utility Target"]


def test_prepared_zone_dt_cont_multiplier_setter_guides_callers_to_problem_api(
    tmp_path: Path,
):
    case_path = copy_sample_case(
        "crude_preheat_train.json",
        tmp_path / "crude_preheat_train.json",
    )
    problem = PinchProblem(source=case_path, project_name=case_path.stem)

    with pytest.raises(RuntimeError, match="PinchProblem.set_dt_cont_multiplier"):
        problem.master_zone.dt_cont_multiplier = 2.0


def test_set_dt_cont_multiplier_rebuilds_default_utilities_and_net_streams():
    payload = {
        "streams": [
            {
                "zone": "Site/AreaA",
                "name": "HotA",
                "t_supply": 180.0,
                "t_target": 120.0,
                "heat_flow": 300.0,
                "dt_cont": 5.0,
                "htc": 1.0,
            }
        ],
        "utilities": [],
        "zone_tree": {
            "name": "Site",
            "type": "Site",
            "children": [{"name": "AreaA", "type": "Process Zone"}],
        },
        "options": {},
    }

    problem = PinchProblem(source=payload, project_name="Site")
    problem.target()

    area = problem.master_zone.get_subzone("AreaA")
    cold_utility = next(
        utility for utility in area.cold_utilities if utility.name == "CU"
    )
    assert cold_utility.dt_cont == pytest.approx(area.config.DT_CONT)
    assert cold_utility.dt_cont_act == pytest.approx(area.config.DT_CONT)
    assert len(area.net_hot_streams) > 0

    problem.set_dt_cont_multiplier(3.0, zone_name="AreaA")
    problem.target()

    area = problem.master_zone.get_subzone("AreaA")
    cold_utility = next(
        utility for utility in area.cold_utilities if utility.name == "CU"
    )
    assert cold_utility.dt_cont == pytest.approx(area.config.DT_CONT)
    assert cold_utility.dt_cont_act == pytest.approx(area.config.DT_CONT * 3.0)
    assert len(area.net_hot_streams) > 0

    net_stream = area.net_hot_streams[0]
    assert net_stream.dt_cont == pytest.approx(cold_utility.dt_cont_act)
    assert net_stream.dt_cont_act == pytest.approx(cold_utility.dt_cont_act)
