"""A design's heat pump is targeted on the prices its networks are ranked on.

The heat pump target prices heat and cooling as ratios of the electricity
price (``COSTING_HPR_PRICE_RATIO_*``), heat at the electricity price by
default; the design ranks its networks at the utilities' own prices. On
the levels case steam costs 40 $/MWh against electricity at 100, so a
heat pump pays only above a COP of 2.5, yet it was sized as if every
machine paid. The design case now prices heat and cooling at the
utilities the direct integration target draws, weighted by their duties.
"""

from __future__ import annotations

import pytest

from OpenPinch.application._problem.accessors import target as target_accessor
from OpenPinch.application.heat_pump_design import heat_pump_design_case
from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    levels_problem,
)


def _options_asked(monkeypatch, problem, **case):
    asked = {}
    original = target_accessor._TargetAccessor.heat_pump_and_refrigeration

    def recording(self, *args, **kwargs):
        asked.update(kwargs.get("options") or {})
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        target_accessor._TargetAccessor, "heat_pump_and_refrigeration", recording
    )
    heat_pump_design_case(
        problem,
        1.0,
        load={"fraction": 1.0},
        levels={"condensers": 1, "evaporators": 1},
        search={"restarts": 0, "evaluations": 1},
        **case,
    )
    return asked


def test_the_heat_pump_is_targeted_at_the_drawn_utilities_prices(monkeypatch):
    asked = _options_asked(monkeypatch, levels_problem())

    # HPS at 40 $/MWh and CW at 5 $/MWh against electricity at 100 $/MWh.
    assert asked.get("COSTING_HPR_PRICE_RATIO_HEAT_TO_ELE") == pytest.approx(0.4)
    assert asked.get("COSTING_HPR_PRICE_RATIO_COOLING_WATER_TO_ELE") == pytest.approx(
        0.05
    )


def test_a_price_ratio_the_user_sets_is_kept(monkeypatch):
    asked = _options_asked(
        monkeypatch,
        levels_problem(),
        options={"COSTING_HPR_PRICE_RATIO_HEAT_TO_ELE": 0.7},
    )

    assert asked["COSTING_HPR_PRICE_RATIO_HEAT_TO_ELE"] == 0.7
    assert asked.get("COSTING_HPR_PRICE_RATIO_COOLING_WATER_TO_ELE") == pytest.approx(
        0.05
    )
