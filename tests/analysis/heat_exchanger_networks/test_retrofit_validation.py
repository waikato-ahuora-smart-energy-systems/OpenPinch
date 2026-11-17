"""Retrofit model checked against Akpomiemie and Smith (2018).

The case is their simplified crude preheat train ("Cost-effective strategy
for heat exchanger network retrofit", Energy, doi 10.1016/j.energy.2017.09.005):
stream data from Table 3, the existing network from Fig. 10 and its duties
from Table 4, hot utility 400 $/kW y and cooling water 5.5 $/kW y (Table 5).

OpenPinch takes film coefficients per stream, not U per exchanger, so every
stream has h = 0.5 kW/m2K (U = 0.25, the paper's U for a new exchanger) and
each installed area is back-calculated from the published duty and
temperatures at that U: U*A, and so the rating, matches the paper. The
utility exchangers have no published area and are given areas that never
bind (the paper's retrofit removes all cooling).
"""

from __future__ import annotations

import json

import pytest

from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
    require_solver_binary,
    require_synthesis_dependency,
)
from OpenPinch.application.problem import PinchProblem
from tests.support.paths import FIXTURES_ROOT

CASE_JSON = FIXTURES_ROOT / "retrofit" / "Six-stream-Akpomiemie-and-Smith-2018.json"

# Table 4: base hot utility (kW).
PUBLISHED_HOT_UTILITY = 14_455.41
# Section 5.2: the existing structure "pinched" (unlimited area) at a 10 K
# approach saves 13.9 % of hot utility; Table 9: the two structural changes
# (Fig. 19) reach 10,958.81 kW, 24.2 %, with every cooler at zero.
PINCHED_SAVING = 0.139
RETROFITTED_HOT_UTILITY = 10_958.81

# Section 5.1 and Tables 6 and 8: twisted-tape inserts raise U on E5 from
# 0.325 to 0.448 and on E1 from 0.517 to 0.669; enhancing both (E5 first)
# brings hot utility to 13,661.52 kW, 5.5 % less. The next candidate, E2,
# breaks C1's pressure drop limit, which OpenPinch does not model, so it is
# left out. Inserts cost 500 + 10 A $ (Table 5).
ENHANCEMENT_FACTORS = {"E5": 0.448 / 0.325, "E1": 0.669 / 0.517}
ENHANCED_HOT_UTILITY = 13_661.52
# Section 5.1: inserts raise C1's tube-side pressure drop in E5 from 15.73
# to 37.67 kPa, in E1 from 167.29 to 169.25 and in E2 (U 0.475 to 0.597,
# Table 6) from 131.40 to 163.31; C1 runs at 668.51 kPa of its 700 kPa
# limit (Table 3), so E2's insert is the one that breaks it.
E2_FACTOR = 0.597 / 0.475
INSERT_PRESSURE_DROPS = {
    "E5": ("C1", 37.67 - 15.73),
    "E1": ("C1", 169.25 - 167.29),
    "E2": ("C1", 163.31 - 131.40),
}
C1_ALLOWANCE = 700.0 - 668.51

# Fig. 10, in stages: C1 from its hot end meets 1, 5, 2, 4, 6, 7, 3; H1 meets
# 1, 2, 3. Areas in m2 at U = 0.25 from the Table 4 duties.
EXISTING = {
    "recovery": [
        {"hot": "H1", "cold": "C1", "stage": 1, "area": 720.56, "id": "E1"},
        {"hot": "H3", "cold": "C1", "stage": 2, "area": 233.72, "id": "E5"},
        {"hot": "H1", "cold": "C1", "stage": 3, "area": 850.05, "id": "E2"},
        {"hot": "H2", "cold": "C1", "stage": 4, "area": 127.92, "id": "E4"},
        {"hot": "H4", "cold": "C1", "stage": 5, "area": 199.23, "id": "E6"},
        {"hot": "H5", "cold": "C1", "stage": 6, "area": 163.22, "id": "E7"},
        {"hot": "H1", "cold": "C1", "stage": 7, "area": 354.22, "id": "E3"},
    ],
    "heaters": [{"stream": "C1", "area": 1.0e4, "id": "H"}],
    "coolers": [
        {"stream": stream, "area": 1.0e4, "id": f"C{n}"}
        for n, stream in enumerate(("H1", "H2", "H3", "H4"), start=1)
    ],
}
# Fig. 19: E7 moved along C1 and a new H1-C1 exchanger N between E2 and E3.
RETROFITTED = {
    "recovery": [
        {"hot": "H1", "cold": "C1", "stage": 1, "id": "E1"},
        {"hot": "H3", "cold": "C1", "stage": 2, "id": "E5"},
        {"hot": "H1", "cold": "C1", "stage": 3, "id": "E2"},
        {"hot": "H5", "cold": "C1", "stage": 4, "id": "E7"},
        {"hot": "H1", "cold": "C1", "stage": 5, "id": "N"},
        {"hot": "H2", "cold": "C1", "stage": 6, "id": "E4"},
        {"hot": "H4", "cold": "C1", "stage": 7, "id": "E6"},
        {"hot": "H1", "cold": "C1", "stage": 8, "id": "E3"},
    ],
    "heaters": [{"stream": "C1", "id": "H"}],
    "coolers": [
        {"stream": stream, "id": f"C{n}"}
        for n, stream in enumerate(("H1", "H2", "H3", "H4"), start=1)
    ],
}


def _live_solver_available() -> bool:
    try:
        require_synthesis_dependency("pyomo.environ", purpose="retrofit validation")
        require_solver_binary("ipopt", purpose="retrofit validation")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


_needs_live_solver = pytest.mark.skipif(
    "not _live_solver_available()",
    reason="needs the ipopt binary and the pyomo package",
)


def _problem() -> PinchProblem:
    return PinchProblem(json.loads(CASE_JSON.read_text(encoding="utf-8")))


def _unlimited(structure):
    """``structure`` without installed areas: duties limited by approach only."""

    return {
        "recovery": [
            {k: v for k, v in e.items() if k != "area"} for e in structure["recovery"]
        ],
        "heaters": [
            {"stream": e["stream"], "id": e["id"]} for e in structure["heaters"]
        ],
        "coolers": [
            {"stream": e["stream"], "id": e["id"]} for e in structure["coolers"]
        ],
    }


@pytest.mark.synthesis
@pytest.mark.solver
@_needs_live_solver
def test_the_existing_network_rates_close_to_its_published_hot_utility():
    result = _problem().design.retrofit(EXISTING, max_modifications=0)

    # The rating may shift duty within the installed areas (2 % tolerance)
    # and finds slightly less hot utility than the published duties.
    assert result.frontier[0].hot_utility == pytest.approx(
        PUBLISHED_HOT_UTILITY, rel=0.01
    )
    assert result.frontier[0].hot_utility <= PUBLISHED_HOT_UTILITY


@pytest.mark.synthesis
@pytest.mark.solver
@_needs_live_solver
def test_the_existing_structure_with_unlimited_area_saves_the_published_share():
    view = _problem().design.optimise_duties(
        _unlimited(EXISTING), objective="utility", min_approach_temperature=10.0
    )

    saving = 1.0 - view.total_hot_utility / PUBLISHED_HOT_UTILITY
    assert saving == pytest.approx(PINCHED_SAVING, abs=0.001)


@pytest.mark.synthesis
@pytest.mark.solver
@_needs_live_solver
def test_the_published_retrofit_structure_reaches_the_published_hot_utility():
    view = _problem().design.optimise_duties(
        RETROFITTED, objective="utility", min_approach_temperature=10.0
    )

    assert view.total_hot_utility == pytest.approx(RETROFITTED_HOT_UTILITY, rel=1e-3)


@pytest.mark.synthesis
@pytest.mark.solver
@_needs_live_solver
def test_the_search_reaches_the_published_retrofit_target():
    # Table 5 costs (one area law at $300/m2), a 10 K approach and every new
    # match position. The paper reaches 24.2 % with two structural changes
    # and about 1,760 m2 on existing units; the search, counting each area
    # addition as a modification, needs seven steps.
    result = _problem().design.retrofit(
        EXISTING,
        strategy="stepwise",
        max_modifications=7,
        new_match_candidates=24,
        min_approach_temperature=10.0,
        area_addition_cost=4_000.0,
        split_cost=10_000.0,
    )

    best = min(design.hot_utility for design in result.frontier)
    assert best == pytest.approx(RETROFITTED_HOT_UTILITY, rel=0.005)


@pytest.mark.synthesis
@pytest.mark.solver
@_needs_live_solver
def test_the_structure_strategy_reaches_the_target_in_three_changes():
    # Ranking structures by their unlimited-area ceiling and sizing area
    # within each solve, three structural changes reach the published hot
    # utility; area additions do not count as steps.
    result = _problem().design.retrofit(
        EXISTING,
        strategy="structure",
        max_modifications=3,
        min_approach_temperature=10.0,
        area_addition_cost=4_000.0,
        split_cost=10_000.0,
    )

    best = result.frontier[3]
    structural = [m for m in best.modifications if m.kind != "area_addition"]
    assert len(structural) == 3
    assert best.hot_utility == pytest.approx(RETROFITTED_HOT_UTILITY, rel=0.005)


@pytest.mark.synthesis
@pytest.mark.solver
@_needs_live_solver
def test_enhancing_e5_then_e1_reaches_the_published_hot_utility():
    # A fixed structure: no new units or added area, inserts only.
    result = _problem().design.retrofit(
        EXISTING,
        max_modifications=2,
        strategy="stepwise",
        new_unit_cost=1e9,
        area_addition_cost=1e9,
        enhancement_factors=ENHANCEMENT_FACTORS,
        enhancement_cost=500.0,
        enhancement_area_cost=10.0,
    )

    steps = [
        [m.exchanger_id for m in design.modifications if m.kind == "enhancement"]
        for design in result.frontier
    ]
    assert steps == [[], ["E5"], ["E5", "E1"]]
    # As for the base rating, the 2 % area tolerance lets the rating find a
    # little less hot utility than the published duties.
    assert result.frontier[2].hot_utility == pytest.approx(
        ENHANCED_HOT_UTILITY, rel=0.01
    )


@pytest.mark.synthesis
@pytest.mark.solver
@_needs_live_solver
def test_the_pressure_drop_limit_rules_out_e2_as_the_paper_does():
    result = _problem().design.retrofit(
        EXISTING,
        max_modifications=3,
        strategy="stepwise",
        new_unit_cost=1e9,
        area_addition_cost=1e9,
        enhancement_factors={**ENHANCEMENT_FACTORS, "E2": E2_FACTOR},
        enhancement_cost=500.0,
        enhancement_area_cost=10.0,
        enhancement_pressure_drops=INSERT_PRESSURE_DROPS,
        pressure_drop_allowances={"C1": C1_ALLOWANCE},
    )

    steps = [
        [m.exchanger_id for m in design.modifications if m.kind == "enhancement"]
        for design in result.frontier
    ]
    assert steps[:3] == [[], ["E5"], ["E5", "E1"]]
    assert not any("E2" in step for step in steps)
