"""Focused record-construction utility branches."""

from __future__ import annotations

import numpy as np

from OpenPinch.analysis.heat_pumps.performance_maps import target_records


def test_distribution_version_handles_missing_distribution(monkeypatch) -> None:
    monkeypatch.setattr(
        target_records,
        "version",
        lambda _name: (_ for _ in ()).throw(target_records.PackageNotFoundError),
    )

    assert target_records._distribution_version("missing") == "unknown"


def test_scalar_property_backend_and_pair_thaw_helpers() -> None:
    assert target_records._first_scalar(None, default=2.0) == 2.0
    assert target_records._first_scalar(np.array([]), default=3.0) == 3.0
    assert target_records._first_scalar(np.array([4.0]), default=0.0) == 4.0
    assert target_records._property_backend("R134a") == "HEOS"
    assert target_records._property_backend("HEOS::R134a") == "HEOS"
    assert target_records._thaw_pairs({"value": 1}) == {"value": 1}
    assert target_records._thaw_pairs(object()) == {}
    assert target_records._thaw_pairs((("nested", (("items", (1, 2)),)),)) == {
        "nested": {"items": [1, 2]}
    }
