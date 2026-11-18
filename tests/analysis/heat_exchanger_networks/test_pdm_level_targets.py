"""Per-level utility targets in the pinch design decomposition.

Each period target carries the duty targeting assigns every hot and cold
utility at the task's dt_cont multiplier, in the solver's utility order, so
the pinch design sides can hold each level at its target.
"""

import json

import pytest

from OpenPinch.analysis.heat_exchanger_networks.solver import (
    pinch_design_decomposition as decomposition_module,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    ZERO_CONTRIBUTION_REPLACEMENT,
)
from OpenPinch.application.case_derivation import _zone_case
from OpenPinch.application.problem import PinchProblem
from tests.support.paths import REPOSITORY_ROOT

ZONAL_SITE = REPOSITORY_ROOT / "OpenPinch/tutorials/sample_cases/zonal_site.json"


def _process(zone: str) -> PinchProblem:
    """One process of ``zonal_site``, as synthesis sees it.

    Synthesis replaces a zero dt_cont with 0.1 K (``ZERO_CONTRIBUTION_REPLACEMENT``),
    and the sample's utilities have zero contributions, so they are set to 0.1 K
    here for targeting to see the same levels.
    """
    source = json.loads(ZONAL_SITE.read_text(encoding="utf-8"))
    for utility in source["utilities"]:
        if utility["dt_cont"]["value"] == 0:
            utility["dt_cont"]["value"] = ZERO_CONTRIBUTION_REPLACEMENT
    site = PinchProblem(source)
    return _zone_case(site, site._resolve_target_zone(zone))


def _targeted(problem: PinchProblem, zone: str) -> dict[str, float]:
    result = problem.target.direct_heat_integration(zone=zone)
    return {
        f"{side}.{utility.name}": float(utility.heat_flow)
        for side, utilities in (
            ("Hot Utility", result.hot_utilities),
            ("Cold Utility", result.cold_utilities),
        )
        for utility in utilities
    }


@pytest.mark.parametrize("zone", ["Hospital[S]", "Food Plant[S]", "Chemical Plant[S]"])
def test_level_targets_are_targetings_duty_for_each_utility(zone):
    problem = _process(zone)
    expected = _targeted(problem, zone)

    decomposition = decomposition_module.build_pinch_design_decomposition(
        problem, 10.0, pinch_location="above"
    )

    (target,) = decomposition.period_targets
    levels = dict(
        zip(decomposition.hot_utility_identities, target.hot_utility_level_targets)
    ) | dict(
        zip(decomposition.cold_utility_identities, target.cold_utility_level_targets)
    )
    assert set(levels) == set(expected)
    for key, duty in expected.items():
        assert levels[key] == pytest.approx(duty, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("dTmin", [5.0, 10.0, 20.0])
def test_level_targets_sum_to_the_utility_targets_at_each_multiplier(dTmin):
    decomposition = decomposition_module.build_pinch_design_decomposition(
        _process("Chemical Plant[S]"), dTmin, pinch_location="below"
    )

    (target,) = decomposition.period_targets
    assert sum(target.hot_utility_level_targets) == pytest.approx(
        target.hot_utility_target, rel=1e-9, abs=1e-9
    )
    assert sum(target.cold_utility_level_targets) == pytest.approx(
        target.cold_utility_target, rel=1e-9, abs=1e-9
    )
