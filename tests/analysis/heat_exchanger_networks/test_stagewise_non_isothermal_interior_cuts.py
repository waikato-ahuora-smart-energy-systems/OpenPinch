"""The non-isothermal stage-wise model checks its matches inside, too.

The isothermal model installs the interior approach cuts when a stream is
segmented: each solution is checked at the segment boundaries inside its
matches. The non-isothermal one did not, so a match on a condensing
stream could cross at the kink; on the condensing stream V against C the
solved match then failed its slice areas ("Segmented exchanger slice has a
non-positive terminal temperature difference").
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.models.stagewise import (
    StageWiseModel,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from tests.analysis.heat_exchanger_networks.test_stagewise_nlp_interior_approach import (
    _kink_problem,
)


def _couenne_available() -> bool:
    # Checked when the test runs, not at import: several workers importing
    # at once race to create the solver's home directory.
    from tests.analysis.heat_exchanger_networks.test_stagewise_nlp_interior_approach import (
        _couenne_available as available,
    )

    return available()


@pytest.mark.solver
@pytest.mark.skipif("not _couenne_available()", reason="needs couenne")
def test_a_non_isothermal_model_on_a_segmented_stream_checks_inside_its_matches():
    model = StageWiseModel(
        name="kink-non-isothermal",
        framework="ESM",
        solver="couenne",
        solver_arrays=problem_to_solver_arrays(_kink_problem(), 2.0),
        stages=1,
        dTmin=2.0,
        z_restriction=None,
        min_dqda=0.0,
        minimisation_goal="hot utility",
        non_isothermal_model=True,
        integers=False,
        tol=1e-6,
    )

    assert hasattr(model, "_add_lazy_cuts")
