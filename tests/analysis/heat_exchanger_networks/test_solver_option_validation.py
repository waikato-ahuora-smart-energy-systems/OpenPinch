"""Solver option names and values are validated before any option file."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.contracts.input import TargetInput
from OpenPinch.domain.configuration import Configuration

OPTION_KEYS = (
    "HENS_SOLVER_OPTIONS_PDM",
    "HENS_SOLVER_OPTIONS_TDM",
    "HENS_SOLVER_OPTIONS_EVM",
)


@pytest.mark.parametrize("key", OPTION_KEYS)
@pytest.mark.parametrize(
    "options",
    [
        {"tol": 1e-6, "max_iter": 500, "print_level": 0},
        {"node_limit": 50, "time_limit": 120.5, "allowable_gap": "1e-4"},
        {"mu_strategy": "adaptive", "delete_redundant": "no"},
        {"minlp_maximum_iterations": 4, "use_warm_start": True},
    ],
)
def test_allow_listed_solver_options_are_accepted(key, options):
    assert Configuration(options={key: options}).hens is not None
    assert TargetInput(streams=[], options={key: options}).options[key] == options


@pytest.mark.parametrize("key", OPTION_KEYS)
@pytest.mark.parametrize(
    "options",
    [
        {"not_a_solver_option": 1},
        {"output_file": "result.txt"},
        {"tol": "1e-6\nmax_iter 5"},
        {"tol": "1e-6 5"},
        {"mu_strategy": "../adaptive"},
        {"mu_strategy": "a/b"},
        {"max_iter": float("nan")},
        {"max_iter": [1, 2]},
        {"max_iter": None},
        {"tol": ""},
        {"max_iter": 10**1000},
        {"max_iter": "9" * 40},
    ],
)
def test_other_solver_option_names_and_values_are_rejected(key, options):
    with pytest.raises(ValueError, match=key):
        Configuration(options={key: options})

    with pytest.raises(ValidationError):
        TargetInput(streams=[], options={key: options})


class _Model:
    def __init__(self, path: Path) -> None:
        self.options = SimpleNamespace(SOLVER=None, SOLVER_EXTENSION=None)
        self.solver_options: list[str] = []
        self._path = str(path)


def _present_solver(monkeypatch) -> None:
    class _Factory:
        def available(self, exception_flag: bool = False) -> bool:
            del exception_flag
            return True

    def dependency(import_name: str, **_kwargs):
        if import_name == "pyomo.environ":
            return SimpleNamespace(SolverFactory=lambda _name: _Factory())
        return object()

    monkeypatch.setattr(
        backend, "require_solver_binary", lambda name, **_kwargs: f"/bin/{name}"
    )
    monkeypatch.setattr(backend, "require_synthesis_dependency", dependency)


@pytest.mark.parametrize(
    ("solver", "option_file"),
    [("ipopt-pyomo", "ipopt.opt"), ("couenne", "couenne.opt")],
)
@pytest.mark.parametrize(
    "solver_options",
    [
        {"not_a_solver_option": 1},
        {"tol": "1e-6\nmax_iter 5"},
        ["tol 1e-6", "not_a_solver_option 1"],
        ["tol"],
    ],
)
def test_option_files_are_written_only_from_validated_options(
    monkeypatch, tmp_path, solver, option_file, solver_options
):
    _present_solver(monkeypatch)
    model = _Model(tmp_path / "run")

    with pytest.raises(ValueError, match="solver option"):
        backend.configure_gekko_solver(model, solver, solver_options=solver_options)

    assert not (tmp_path / "run" / option_file).exists()


@pytest.mark.parametrize("solver", ["ipopt-GEKKO", "apopt"])
def test_gekko_solver_option_lines_are_validated(monkeypatch, tmp_path, solver):
    _present_solver(monkeypatch)

    with pytest.raises(ValueError, match="solver option"):
        backend.configure_gekko_solver(
            _Model(tmp_path / "run"),
            solver,
            solver_options={"not_a_solver_option": 1},
        )


def test_option_file_writer_rejects_unvalidated_options(tmp_path):
    with pytest.raises(ValueError, match="solver option"):
        backend._write_solver_option_file(
            _Model(tmp_path / "run"), "ipopt.opt", {"not_a_solver_option": 1}
        )

    assert not (tmp_path / "run" / "ipopt.opt").exists()


def test_valid_options_are_written_to_the_option_file(monkeypatch, tmp_path):
    _present_solver(monkeypatch)
    model = _Model(tmp_path / "run")

    run = backend.configure_gekko_solver(
        model,
        "ipopt-pyomo",
        solver_options={"tol": 1e-6, "max_iter": 50, "mu_strategy": "adaptive"},
    )

    assert Path(run.option_file).read_text(encoding="utf-8") == (
        "tol 1e-06\nmax_iter 50\nmu_strategy adaptive\n"
    )
