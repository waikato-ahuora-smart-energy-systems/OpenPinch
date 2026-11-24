"""The hand-over to direct evolution holds at search depth 0 too.

Depth 0 is pinch design alone. When the forbidden matches leave it no
network there is nothing else to return, so the run evolves directly at
that depth as well, rather than fail.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.execution.settings import (
    workflow_settings_from_problem,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from OpenPinch.analysis.heat_exchanger_networks.targeting.open_hens_method import (
    execute_open_hens_method,
)
from tests.analysis.heat_exchanger_networks.test_forbidden_matches_fallback import (
    _NoPinchDesign,
)
from tests.analysis.heat_exchanger_networks.test_full_sequence import (
    _small_problem,
    _small_problem_with_options,
)


def test_depth_zero_evolves_directly_when_pinch_design_finds_nothing():
    arrays = problem_to_solver_arrays(_small_problem(), 10.0)
    pair = [
        arrays.stream_identities["hot_process_streams"][0],
        arrays.stream_identities["cold_process_streams"][0],
    ]
    problem = _small_problem_with_options(
        {"HENS_FORBIDDEN_MATCHES": [pair], "HENS_SEARCH_DEPTH": 0}
    )
    settings = workflow_settings_from_problem(problem)
    assert not settings.runs_evolution

    with pytest.warns(UserWarning, match="HENS_FORBIDDEN_MATCHES"):
        result = execute_open_hens_method(problem, settings, executor=_NoPinchDesign())

    assert result.accepted_result.network is not None
    assert result.accepted_result.method == "network_evolution_method"
