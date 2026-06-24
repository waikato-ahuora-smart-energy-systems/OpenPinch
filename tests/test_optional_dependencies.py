"""Tests for optional dependency guidance."""

from __future__ import annotations

import pytest

from OpenPinch.services.heat_pump_integration.common._shared import plotting
from OpenPinch.utils.optional_dependencies import optional_dependency_error


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
    monkeypatch.setattr(plotting, "_PLOTLY_IMPORT_ERROR", ImportError("missing"))

    with pytest.raises(ImportError) as exc_info:
        plotting._require_plotly()

    message = str(exc_info.value)
    assert "openpinch[notebook]" in message
    assert "openpinch[dashboard]" in message
