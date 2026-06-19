"""Optional dependency guards for internal HEN synthesis code."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from shutil import which
from types import ModuleType

SYNTHESIS_EXTRA = "synthesis"


class MissingSynthesisDependencyError(ImportError):
    """Raised when the optional synthesis extra is needed but unavailable."""


class MissingSynthesisSolverError(RuntimeError):
    """Raised when an external solver executable is needed but unavailable."""


@dataclass(frozen=True)
class SynthesisDependency:
    """One optional Python package used by future HEN synthesis internals."""

    package: str
    import_name: str
    purpose: str


SYNTHESIS_DEPENDENCIES: tuple[SynthesisDependency, ...] = (
    SynthesisDependency("pyomo", "pyomo.environ", "Pyomo model construction"),
    SynthesisDependency("gekko", "gekko", "GEKKO model construction"),
    SynthesisDependency("matplotlib", "matplotlib", "diagnostic plotting"),
    SynthesisDependency("plotly", "plotly", "interactive HEN plots"),
    SynthesisDependency("kaleido", "kaleido", "static plot export"),
    SynthesisDependency("openpyxl", "openpyxl", "workbook export"),
    SynthesisDependency("wakepy", "wakepy", "long local solver runs"),
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
            "HEN synthesis dependencies with "
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
    path = which(binary_name)
    if path is not None:
        return path

    purpose_text = f" for {purpose}" if purpose else ""
    raise MissingSynthesisSolverError(
        f"The {binary_name!r} solver executable is required{purpose_text}, but "
        "it was not found on PATH. Install the solver binary, confirm it is "
        "available on PATH, and see the OpenPinch synthesis dependency policy "
        "documentation for solver-test requirements."
    )
