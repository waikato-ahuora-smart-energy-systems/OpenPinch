"""Presentation metadata for editable OpenPinch configuration."""

from __future__ import annotations


def configuration_options():
    """Return report-friendly metadata for supported configuration fields."""
    from ..application._workspace.views.input import configuration_field_metadata

    return configuration_field_metadata()


__all__ = ["configuration_options"]
