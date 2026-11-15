"""Optional dependency guards for internal heat exchanger network synthesis code."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from shutil import which
from types import ModuleType

SYNTHESIS_EXTRA = "synthesis"


class MissingSynthesisDependencyError(ImportError):
    """Raised when the optional synthesis extra is needed but unavailable."""


class MissingSynthesisSolverError(RuntimeError):
    """Raised when an external solver executable is needed but unavailable."""


@dataclass(frozen=True)
class SynthesisDependency:
    """One optional package used by synthesis internals."""

    package: str
    import_name: str
    purpose: str


SYNTHESIS_DEPENDENCIES: tuple[SynthesisDependency, ...] = (
    SynthesisDependency("pyomo", "pyomo.environ", "Pyomo model construction"),
    SynthesisDependency("plotly", "plotly", "interactive heat exchanger network plots"),
    SynthesisDependency("kaleido", "kaleido", "static plot export"),
    SynthesisDependency("openpyxl", "openpyxl", "workbook export"),
    SynthesisDependency("wakepy", "wakepy", "long local solver runs"),
    SynthesisDependency("pyscipopt", "pyscipopt", "SCIP MINLP solves"),
    SynthesisDependency("openpinch-solvers", "openpinch_solvers", "Ipopt executable"),
)


def require_synthesis_dependency(
    import_name: str,
    *,
    package: str | None = None,
    purpose: str | None = None,
) -> ModuleType:
    """Import one optional synthesis dependency or raise an actionable error."""
    package_name = package or import_name.split(".", maxsplit=1)[0]
    purpose_text = f" for {purpose}" if purpose else ""

    try:
        return import_module(import_name)
    except ImportError as exc:
        raise MissingSynthesisDependencyError(
            f"{package_name} is required{purpose_text}. Install the optional "
            "heat exchanger network synthesis dependencies with "
            f"'python -m pip install \"openpinch[{SYNTHESIS_EXTRA}]\"'. "
            "See the OpenPinch synthesis dependency policy documentation for "
            "the optional install and test requirements."
        ) from exc


def require_declared_synthesis_dependencies() -> dict[str, ModuleType]:
    """Import the declared synthesis dependency set for install smoke checks."""
    return {
        dependency.package: require_synthesis_dependency(
            dependency.import_name,
            package=dependency.package,
            purpose=dependency.purpose,
        )
        for dependency in SYNTHESIS_DEPENDENCIES
    }


def require_solver_binary(binary_name: str, *, purpose: str | None = None) -> str:
    """Return a solver executable path or raise an actionable error."""
    path = which(binary_name, path=_path_without_imported_idaes())
    if path is not None:
        return path
    package_path = _solver_binary_in(_openpinch_solvers_bin_directory(), binary_name)
    if package_path is not None:
        _prepend_path(package_path.parent)
        return str(package_path)
    package_path = _solver_binary_in(_idaes_bin_directory(), binary_name)
    if package_path is not None:
        # Last on PATH, so IDAES's older Ipopt never shadows another one.
        _append_path(package_path.parent)
        return str(package_path)

    purpose_text = f" for {purpose}" if purpose else ""
    raise MissingSynthesisSolverError(
        f"The {binary_name!r} solver executable is required{purpose_text}, but "
        "it was not found on PATH. Install the solver binary, confirm it is "
        "available on PATH, and see the OpenPinch synthesis dependency policy "
        "documentation for solver-test requirements."
    )


def _path_without_imported_idaes() -> str:
    """``PATH`` without the IDAES download that importing ``idaes`` puts first.

    That directory stays a fallback after ``openpinch_solvers``; otherwise
    looking for Couenne would switch every later Ipopt solve to IDAES's
    older Ipopt.
    """

    entries = os.environ.get("PATH", "").split(os.pathsep)
    idaes_bin = getattr(sys.modules.get("idaes"), "bin_directory", None)
    if idaes_bin is None:
        return os.pathsep.join(entries)
    return os.pathsep.join(entry for entry in entries if entry != str(idaes_bin))


def _solver_binary_in(bin_directory: Path | None, binary_name: str) -> Path | None:
    if bin_directory is None:
        return None
    for name in (binary_name, f"{binary_name}.exe"):
        path = bin_directory / name
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


def _openpinch_solvers_bin_directory() -> Path | None:
    """The ``openpinch-solvers`` wheel's executables (Ipopt), if installed."""

    return _package_bin_directory("openpinch_solvers")


def _idaes_bin_directory() -> Path | None:
    """IDAES's downloaded solver executables (Ipopt, Couenne), if installed.

    Importing ``idaes`` puts that directory first on ``PATH``; ``PATH`` is
    restored, so worker processes, which inherit it, do not run IDAES's Ipopt.
    """

    path = os.environ.get("PATH")
    try:
        return _package_bin_directory("idaes")
    finally:
        if path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = path


def _package_bin_directory(module_name: str) -> Path | None:
    try:
        module = import_module(module_name)
    except ImportError:
        return None
    raw_path = getattr(module, "bin_directory", None)
    if raw_path is None:
        return None
    return Path(raw_path)


def _append_path(directory: Path) -> None:
    directory_text = str(directory)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if directory_text not in path_entries:
        os.environ["PATH"] = os.pathsep.join([*path_entries, directory_text])


def _prepend_path(directory: Path) -> None:
    directory_text = str(directory)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if directory_text not in path_entries:
        os.environ["PATH"] = os.pathsep.join([directory_text, *path_entries])
