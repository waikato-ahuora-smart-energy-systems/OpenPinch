"""Coverage tests for repository-level entrypoint scripts."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import OpenPinch


REPO_ROOT = Path(__file__).resolve().parents[1]


class _DummyPinchProblem:
    created = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.calls = []
        _DummyPinchProblem.created.append(self)

    def load(self, path):
        self.calls.append(("load", str(path)))

    def target(self):
        self.calls.append(("target", None))

    def export_to_Excel(self, path):
        self.calls.append(("export_to_Excel", str(path)))

    def render_streamlit_dashboard(self):
        self.calls.append(("render_streamlit_dashboard", None))


class _FakeStreamlit:
    def __init__(self, *, stop_exc: Exception | None = None):
        self.stop_exc = stop_exc or RuntimeError("streamlit stop")
        self.errors = []

    def cache_resource(self, fn):
        return fn

    def error(self, msg):
        self.errors.append(str(msg))

    def stop(self):
        raise self.stop_exc


def _clear_dummy():
    _DummyPinchProblem.created.clear()


def test_root_init_script_executes():
    namespace = runpy.run_path(str(REPO_ROOT / "__init__.py"))
    assert "PinchProblem" in namespace


def test_repo_main_script_executes(monkeypatch):
    _clear_dummy()
    monkeypatch.setattr(OpenPinch, "PinchProblem", _DummyPinchProblem)

    runpy.run_path(str(REPO_ROOT / "__main__.py"), run_name="__main__")

    assert len(_DummyPinchProblem.created) == 1
    calls = _DummyPinchProblem.created[0].calls
    assert calls[0][0] == "load"
    assert calls[1][0] == "target"
    assert calls[2][0] == "export_to_Excel"


def test_run_py_script_executes(monkeypatch):
    _clear_dummy()
    monkeypatch.setattr(OpenPinch, "PinchProblem", _DummyPinchProblem)

    runpy.run_path(str(REPO_ROOT / "run.py"), run_name="__main__")

    assert len(_DummyPinchProblem.created) == 1
    calls = _DummyPinchProblem.created[0].calls
    assert calls[0][0] == "load"
    assert calls[1][0] == "target"
    assert calls[2][0] == "export_to_Excel"


def test_streamlit_module_helper_functions(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.setattr(OpenPinch, "PinchProblem", _DummyPinchProblem)

    ns = runpy.run_path(str(REPO_ROOT / "streamlit_app.py"))

    _clear_dummy()
    problem = ns["_load_problem"]("dummy-path")
    assert isinstance(problem, _DummyPinchProblem)
    assert problem.args == ("dummy-path",)
    assert problem.kwargs == {"run": True}

    ns["validate_problem_path"](SimpleNamespace(exists=lambda: True))
    assert fake_st.errors == []


def test_streamlit_validate_problem_path_stops_when_missing(monkeypatch):
    fake_st = _FakeStreamlit(stop_exc=RuntimeError("stop-called"))
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    ns = runpy.run_path(str(REPO_ROOT / "streamlit_app.py"))

    with pytest.raises(RuntimeError, match="stop-called"):
        ns["validate_problem_path"](SimpleNamespace(exists=lambda: False))

    assert fake_st.errors
    assert "Problem file not found" in fake_st.errors[0]


def test_streamlit_app_main_block_executes(monkeypatch):
    _clear_dummy()
    fake_st = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.setattr(OpenPinch, "PinchProblem", _DummyPinchProblem)
    monkeypatch.setattr(Path, "exists", lambda self: True)

    runpy.run_path(str(REPO_ROOT / "streamlit_app.py"), run_name="__main__")

    assert len(_DummyPinchProblem.created) == 1
    calls = _DummyPinchProblem.created[0].calls
    assert calls == [("render_streamlit_dashboard", None)]
