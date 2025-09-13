from OpenPinch.classes.pinch_problem import PinchProblem 

import json
from pathlib import Path
# import types
import sys
import pytest


@pytest.fixture
def sample_problem():
    return {
        "options": {"dt_min": 10},
        "streams": [{"name": "H1", "supply_T": 150, "target_T": 60, "cp": 2.0}],
        "utilities": [{"name": "LP Steam", "T": 150, "cost": 20}],
    }


def test_load_json(tmp_path: Path, sample_problem):
    p = tmp_path / "problem.json"
    p.write_text(json.dumps(sample_problem), encoding="utf-8")

    obj = PinchProblem(run=False)
    out = obj.load(p)

    assert out == sample_problem
    assert obj.problem_filepath == p
    assert obj.results is None
    # to_problem_json should mirror problem_data
    assert obj.to_problem_json() == sample_problem


def test_load_excel_calls_reader_and_sets_path(monkeypatch, tmp_path: Path, sample_problem):
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

    obj = PinchProblem(run=False)
    out = obj.load(x)

    assert out == sample_problem
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

    s = tmp_path / "streams.csv"; s.write_text("name,supply_T,target_T,cp\n", encoding="utf-8")
    u = tmp_path / "utilities.csv"; u.write_text("name,T,cost\n", encoding="utf-8")

    obj = PinchProblem(run=False)
    out = obj.load((s, u))

    assert out == sample_problem
    # CSV tuple is "not a single-file source"
    assert obj.problem_filepath is None
    assert called["args"][0] == s and called["args"][1] == u and called["args"][2] is None


def test_load_csv_directory_bundle(monkeypatch, tmp_path: Path, sample_problem):
    mod = sys.modules[PinchProblem.__module__]
    called = {}

    def fake_csv_reader(streams_csv, utilities_csv, output_json):
        called["args"] = (streams_csv, utilities_csv, output_json)
        return sample_problem

    monkeypatch.setattr(mod, "get_problem_from_csv", fake_csv_reader, raising=True)

    d = tmp_path / "bundle"
    d.mkdir()
    (d / "streams.csv").write_text("name,supply_T,target_T,cp\n", encoding="utf-8")
    (d / "utilities.csv").write_text("name,T,cost\n", encoding="utf-8")

    obj = PinchProblem(run=False)
    out = obj.load(d)

    assert out == sample_problem
    assert obj.problem_filepath == d
    s_csv, u_csv, out_json = called["args"]
    assert s_csv == d / "streams.csv"
    assert u_csv == d / "utilities.csv"
    assert out_json is None


def test_load_csv_directory_missing_files_raises(tmp_path: Path):
    d = tmp_path / "bundle"; d.mkdir()
    # intentionally missing utilities.csv
    (d / "streams.csv").write_text("", encoding="utf-8")

    obj = PinchProblem(run=False)
    with pytest.raises(FileNotFoundError):
        obj.load(d)


def test_load_unrecognized_source_raises(tmp_path: Path):
    weird = tmp_path / "problem.txt"
    weird.write_text("not valid", encoding="utf-8")

    obj = PinchProblem(run=False)
    with pytest.raises(ValueError):
        obj.load(weird)


def test_target_raises_without_problem_loaded():
    obj = PinchProblem(run=False)
    with pytest.raises(RuntimeError):
        obj.target()


def test_export_without_results_dir_raises(sample_problem):
    obj = PinchProblem(run=False)
    # Simulate loaded data so export triggers the "no results_dir" error path
    obj._problem_data = sample_problem
    obj._results = {"ok": True}
    with pytest.raises(ValueError):
        obj.export(None)


def test_export_calls_writer_with_results(monkeypatch, tmp_path: Path):
    mod = sys.modules[PinchProblem.__module__]
    called = {}

    def fake_writer(results, results_dir):
        called["args"] = (results, results_dir)
        output_path = Path(results_dir) / "targets.xlsx"
        # we don't actually write, just return a path
        return output_path

    monkeypatch.setattr(mod, "export_target_summary_to_excel_with_units", fake_writer, raising=True)

    obj = PinchProblem(run=False)
    obj._results = {"foo": "bar"}  # Pretend targeting already ran
    dest = tmp_path / "out"

    path = obj.export(dest)

    assert path == dest / "targets.xlsx"
    assert called["args"][0] == {"foo": "bar"}
    assert called["args"][1] == dest
    # object state updated
    assert obj.results_dir == dest


def test_to_problem_json_without_data_raises():
    obj = PinchProblem(run=False)
    with pytest.raises(RuntimeError):
        obj.to_problem_json()


def test_repr_changes_with_state(tmp_path: Path):
    obj = PinchProblem(run=False)
    r = repr(obj)
    assert "<in-memory or CSV tuple>" in r
    assert "export=<unset>" in r
    assert "results=no" in r

    # set paths / results and check
    obj.problem_filepath = tmp_path / "p.json"
    obj.results_dir = tmp_path / "out"
    obj._results = {}
    r2 = repr(obj)
    assert str(obj.problem_filepath) in r2
    assert str(obj.results_dir) in r2
    assert "results=yes" in r2


def test_from_json_builds_and_roundtrips(sample_problem):
    obj = PinchProblem.from_json(sample_problem)
    assert obj.to_problem_json() == sample_problem