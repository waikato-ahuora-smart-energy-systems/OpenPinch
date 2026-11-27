"""A retrofit says that it does not apply ``HENS_FORBIDDEN_MATCHES``.

Synthesis keeps the pairs the option names out of a network. The retrofit
search does not read it: a new match or a heat pump step may pair streams
it forbids, and nothing said so.
"""

from __future__ import annotations

import pytest

from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    EXISTING,
    levels_problem,
    solvers_available,
)


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not solvers_available(), reason="needs ipopt")
def test_a_retrofit_warns_that_forbidden_matches_are_not_applied():
    with pytest.warns(UserWarning, match="HENS_FORBIDDEN_MATCHES is not applied"):
        levels_problem().design.retrofit(
            EXISTING,
            max_modifications=0,
            options={"HENS_FORBIDDEN_MATCHES": [["H1", "C1"]]},
        )
