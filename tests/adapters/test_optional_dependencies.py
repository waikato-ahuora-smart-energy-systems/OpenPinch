"""Tests for optional dependency guidance."""

from __future__ import annotations

import builtins

import pytest

from OpenPinch.adapters.optional_dependencies import optional_dependency_error
from OpenPinch.analysis.heat_pumps.common._shared import plotting


def test_optional_dependency_error_names_openpinch_extras():
    message = optional_dependency_error(
        package="Plotly",
        purpose="graph rendering",
        extras=("notebook", "dashboard"),
        docs="the graphing guide",
    )

    assert 'python -m pip install "openpinch[notebook]"' in message
    assert 'python -m pip install "openpinch[dashboard]"' in message
    assert "the graphing guide" in message


def test_hpr_plotting_dependency_guard_names_expected_extras(monkeypatch):
    real_import = builtins.__import__

    def import_without_plotly(name, *args, **kwargs):
        if name.split(".", 1)[0] == "plotly":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_plotly)

    with pytest.raises(ImportError) as exc_info:
        plotting._require_plotly()

    message = str(exc_info.value)
    assert "openpinch[notebook]" in message
    assert "openpinch[dashboard]" in message
