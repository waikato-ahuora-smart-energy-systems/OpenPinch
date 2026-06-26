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

    def export_excel(self, path):
        self.calls.append(("export_excel", str(path)))

    def show_dashboard(self):
        self.calls.append(("show_dashboard", None))


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


def test_build_dist_script_help_executes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["python", "--help"])
    namespace = runpy.run_path(str(REPO_ROOT / "scripts" / "build_dist.py"))

    with pytest.raises(SystemExit, match="0"):
        namespace["main"](["--help"])


def test_build_dist_uses_isolation_when_backend_is_missing(monkeypatch):
    namespace = runpy.run_path(str(REPO_ROOT / "scripts" / "build_dist.py"))
    build_command = namespace["_build_command"]

    def fake_find_spec(name: str):
        if name == "build":
            return object()
        if name == "hatchling.build":
            return None
        raise AssertionError(f"unexpected module lookup: {name}")

    monkeypatch.setitem(build_command.__globals__, "find_spec", fake_find_spec)

    command = build_command(Path("/tmp/dist"))

    assert command is not None
    assert command[:5] == [sys.executable, "-m", "build", "--wheel", "--sdist"]
    assert "--no-isolation" not in command
    assert command[-2:] == ["--outdir", "/tmp/dist"]


def test_build_dist_cleanup_only_removes_owned_artifacts(tmp_path):
    namespace = runpy.run_path(str(REPO_ROOT / "scripts" / "build_dist.py"))
    clean_output_dir = namespace["_clean_output_dir"]

    stale_wheel = tmp_path / "openpinch-0.1.0-py3-none-any.whl"
    stale_sdist = tmp_path / "openpinch-0.1.0.tar.gz"
    stale_attestation = tmp_path / "openpinch-0.1.0.tar.gz.publish.attestation"
    unrelated_file = tmp_path / "notes.txt"
    unrelated_dir = tmp_path / "shared"
    unrelated_nested_file = unrelated_dir / "keep.txt"

    stale_wheel.write_text("stale wheel", encoding="utf-8")
    stale_sdist.write_text("stale sdist", encoding="utf-8")
    stale_attestation.write_text("stale attestation", encoding="utf-8")
    unrelated_file.write_text("do not delete", encoding="utf-8")
    unrelated_dir.mkdir()
    unrelated_nested_file.write_text("do not delete", encoding="utf-8")

    clean_output_dir(tmp_path)

    assert not stale_wheel.exists()
    assert not stale_sdist.exists()
    assert not stale_attestation.exists()
    assert unrelated_file.read_text(encoding="utf-8") == "do not delete"
    assert unrelated_nested_file.read_text(encoding="utf-8") == "do not delete"


def test_optional_install_smoke_script_help_executes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["python", "--help"])
    namespace = runpy.run_path(str(REPO_ROOT / "scripts" / "optional_install_smoke.py"))

    with pytest.raises(SystemExit, match="0"):
        namespace["main"](["--help"])


def test_streamlit_module_helper_functions(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.setattr(OpenPinch, "PinchProblem", _DummyPinchProblem)

    ns = runpy.run_path(str(REPO_ROOT / "streamlit_app.py"))

    _clear_dummy()
    problem = ns["_load_problem"]("dummy-path")
    assert isinstance(problem, _DummyPinchProblem)
    assert problem.args == ()
    assert problem.kwargs == {"source": "dummy-path"}
    assert problem.calls == [("target", None)]

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
    assert calls == [("target", None), ("show_dashboard", None)]
