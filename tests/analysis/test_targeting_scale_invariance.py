"""Energy, area and unit targets under heat-flow scaling and temperature shift."""

from __future__ import annotations

import copy
import json
import warnings

import pytest

from OpenPinch import PinchProblem
from OpenPinch.resources import read_sample_case

CASES = [
    "Four-stream-Yee-and-Grossmann-1990-1.json",
    "basic_pinch.json",
    "chocolate_factory.json",
    "process_mvr.json",
    "pulp_mill.json",
]


def _scaled(data: dict, factor: float) -> dict:
    data = copy.deepcopy(data)
    for record in data["streams"] + data.get("utilities", []):
        for key in ("heat_flow", "heat_capacity_flowrate"):
            value = record.get(key)
            if isinstance(value, dict) and value.get("value") is not None:
                value["value"] *= factor
            elif isinstance(value, (int, float)):
                record[key] = value * factor
    return data


def _shifted(data: dict, delta: float) -> dict:
    data = copy.deepcopy(data)
    for record in data["streams"] + data.get("utilities", []):
        for key in ("t_supply", "t_target"):
            value = record.get(key)
            if isinstance(value, dict) and value.get("value") is not None:
                value["value"] += delta
            elif isinstance(value, (int, float)):
                record[key] = value + delta
    return data


def _targets(data: dict):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return PinchProblem(
            data, project_name="Site"
        ).target.heat_exchanger_area_and_cost()


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("factor", [1e-6, 1e-3, 1e3])
def test_targets_scale_with_heat_flows(case, factor):
    # Tolerances that were absolute kW made small problems drift (duties off
    # by up to 12 % at x1e-6) and large ones raise. Heat flows and area scale
    # with the factor; temperatures and unit counts do not.
    data = json.loads(read_sample_case(case))
    base = _targets(data)
    scaled = _targets(_scaled(data, factor))

    for name in ("hot_utility_target", "cold_utility_target", "area"):
        assert float(getattr(scaled, name)) == pytest.approx(
            float(getattr(base, name)) * factor, rel=1e-6
        ), name
    assert scaled.hot_pinch == pytest.approx(base.hot_pinch, abs=1e-6)
    assert scaled.cold_pinch == pytest.approx(base.cold_pinch, abs=1e-6)
    assert scaled.num_units == base.num_units


@pytest.mark.parametrize("case", CASES)
def test_targets_are_unchanged_by_a_uniform_temperature_shift(case):
    data = json.loads(read_sample_case(case))
    base = _targets(data)
    shifted = _targets(_shifted(data, 100.0))

    for name in ("hot_utility_target", "cold_utility_target", "area"):
        assert float(getattr(shifted, name)) == pytest.approx(
            float(getattr(base, name)), rel=1e-6
        ), name
    assert shifted.hot_pinch == pytest.approx(base.hot_pinch + 100.0, abs=1e-6)
    assert shifted.cold_pinch == pytest.approx(base.cold_pinch + 100.0, abs=1e-6)
    assert shifted.num_units == base.num_units
