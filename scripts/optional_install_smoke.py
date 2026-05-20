"""Smoke checks for the published optional install surfaces."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path


def _assert_module_present(module_name: str) -> None:
    __import__(module_name)


def _assert_module_absent(module_name: str) -> None:
    try:
        __import__(module_name)
    except ImportError:
        return
    raise AssertionError(
        f"Optional dependency {module_name!r} should not be installed for the core "
        "surface."
    )


def _exercise_cli_help() -> None:
    import OpenPinch.__main__ as cli

    try:
        cli.main(["notebook", "--help"])
    except SystemExit as exc:
        if exc.code != 0:
            raise AssertionError(
                f"openpinch notebook --help exited with {exc.code!r}"
            ) from exc
    else:
        raise AssertionError("CLI help should terminate via SystemExit.")


def _check_core_surface() -> None:
    import OpenPinch
    from OpenPinch import PinchProblem
    from OpenPinch.classes.brayton_heat_pump import SimpleBraytonHeatPumpCycle
    from OpenPinch.streamlit_webviewer import web_graphing as wg
    from OpenPinch.utils.miscellaneous import graph_simple_cc_plot

    assert OpenPinch.__version__ if hasattr(OpenPinch, "__version__") else True
    assert PinchProblem is not None
    _exercise_cli_help()

    for module_name in [
        "streamlit",
        "plotly",
        "openpyxl",
        "pyxlsb",
        "tespy",
        "ipykernel",
        "nbformat",
    ]:
        _assert_module_absent(module_name)

    try:
        wg._require_streamlit()
    except ImportError:
        pass
    else:
        raise AssertionError("core install should not expose Streamlit.")

    try:
        graph_simple_cc_plot([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    except ImportError:
        pass
    else:
        raise AssertionError("core install should not expose Plotly helpers.")

    try:
        SimpleBraytonHeatPumpCycle()
    except ImportError:
        pass
    else:
        raise AssertionError("core install should not expose TESPy tooling.")


def _check_dashboard_surface() -> None:
    from OpenPinch.streamlit_webviewer import web_graphing as wg

    _exercise_cli_help()
    for module_name in ["streamlit", "plotly", "openpyxl", "pyxlsb"]:
        _assert_module_present(module_name)

    assert wg._require_plotly() is not None
    assert wg._require_streamlit() is not None
    assert wg._require_openpyxl() is not None


def _check_notebook_surface() -> None:
    import nbformat

    from OpenPinch.resources import copy_notebook, list_notebooks
    from OpenPinch.utils.miscellaneous import _require_plotly

    _exercise_cli_help()
    for module_name in ["plotly", "openpyxl", "pyxlsb", "ipykernel", "nbformat"]:
        _assert_module_present(module_name)

    assert _require_plotly() is not None

    notebook_name = list_notebooks()[0]
    with tempfile.TemporaryDirectory() as tmp_dir:
        notebook_path = copy_notebook(notebook_name, Path(tmp_dir) / notebook_name)
        with notebook_path.open("r", encoding="utf-8") as handle:
            notebook = nbformat.read(handle, as_version=4)
    assert notebook["nbformat"] == 4


def _check_brayton_cycle_surface() -> None:
    from OpenPinch.classes.brayton_heat_pump import SimpleBraytonHeatPumpCycle

    _exercise_cli_help()
    _assert_module_present("tespy")
    assert SimpleBraytonHeatPumpCycle() is not None


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for smoke checks."""
    parser = argparse.ArgumentParser(
        prog="optional_install_smoke.py",
        description="Smoke-check the current OpenPinch install surface.",
    )
    parser.add_argument(
        "surface",
        choices=["core", "dashboard", "notebook", "brayton_cycle"],
        help="Install surface to verify.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the selected smoke-check surface."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.surface == "core":
        _check_core_surface()
    elif args.surface == "dashboard":
        _check_dashboard_surface()
    elif args.surface == "notebook":
        _check_notebook_surface()
    elif args.surface == "brayton_cycle":
        _check_brayton_cycle_surface()

    print(f"OpenPinch optional install smoke passed for: {args.surface}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
