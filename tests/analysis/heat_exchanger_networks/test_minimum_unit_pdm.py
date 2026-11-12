"""Minimum-unit networks from the pinch design method.

Each pinch side is re-solved for the fewest units at its utility target, and
no-good cuts collect further minimum-unit and one-more-unit topologies. Side
pairs become extra networks at each approach temperature, all seeds for
network evolution.
"""

from __future__ import annotations

import json

import pytest

from OpenPinch.analysis.heat_exchanger_networks.models.problem import (
    InternalHeatExchangerNetworkProblem,
)
from OpenPinch.analysis.heat_exchanger_networks.reporting.ranking import (
    network_structure_signature,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.pinch_design_decomposition import (
    build_pinch_design_decomposition,
)
from OpenPinch.application.problem import PinchProblem
from tests.support.paths import REPOSITORY_ROOT

FOUR_STREAM_JSON = (
    REPOSITORY_ROOT
    / "tests/fixtures/openhens/Four-stream-Yee-and-Grossmann-1990-1.json"
)

pytestmark = pytest.mark.synthesis


def _highs_available() -> bool:
    try:
        backend.require_solver_backend("highs")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def _problem(dTmin: float, limits: tuple[int, int]):
    pinch = PinchProblem(
        source=json.loads(FOUR_STREAM_JSON.read_text(encoding="utf-8")),
        project_name="Four-stream",
    )
    return InternalHeatExchangerNetworkProblem(
        solver_arrays=problem_to_solver_arrays(pinch, dTmin),
        framework="PDM",
        solver="highs",
        dTmin=dTmin,
        pinch_decompositions={
            side: build_pinch_design_decomposition(pinch, dTmin, pinch_location=side)
            for side in ("above", "below")
        },
        pdm_network_limits=limits,
    )


def _units(network) -> int:
    return len(network.exchangers)


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_pdm_collects_distinct_minimum_unit_networks_as_variants():
    problem = _problem(14.0, (3, 3))

    solved = problem.get_solution(print_output=False)

    assert solved is not None and solved.mSuccess == 1
    networks = [problem.extract_network(run_id="pdm")] + [
        variant.extract_network(run_id="pdm") for variant in problem.pdm_variants
    ]
    assert len(problem.pdm_variants) >= 2
    assert all(variant.case.mSuccess == 1 for variant in problem.pdm_variants)
    signatures = [network_structure_signature(network) for network in networks]
    assert len(set(signatures)) == len(signatures)
    variant_units = [_units(network) for network in networks[1:]]
    assert max(variant_units) - min(variant_units) <= 1
    assert len(variant_units) <= 6


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_pdm_without_network_limits_has_no_variants():
    problem = _problem(14.0, (0, 0))

    problem.get_solution(print_output=False)

    assert problem.pdm_variants == []
