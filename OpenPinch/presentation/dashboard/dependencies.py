"""Optional dependency guards for web graphing."""

from __future__ import annotations

import sys

from ...adapters.optional_dependencies import optional_dependency_error


def _require_streamlit():
    streamlit_mod = sys.modules.get("streamlit")
    if streamlit_mod is not None:
        return streamlit_mod

    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError(
            optional_dependency_error(
                package="Streamlit",
                purpose="render_streamlit_dashboard",
                extras="dashboard",
                docs="the first-solve Python guide",
            )
        ) from exc
    else:
        return st


def _require_openpyxl():
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError(
            optional_dependency_error(
                package="OpenPyXL",
                purpose="dashboard Excel downloads",
                extras=("dashboard", "notebook"),
                docs="the exporting results guide",
            )
        ) from exc
    return openpyxl
