"""A sliced utility's slices keep its forbidden matches.

Where utility levels are held, a sliding utility is cut into slices that
take its place on the utility axis. A forbidden heater or cooler names the
utility, so every slice of it is forbidden on that stream.
"""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.utility_slices import (
    with_utility_slices,
)
from tests.analysis.heat_exchanger_networks.test_staged_utilities import (
    D_TMIN,
    _heating_problem,
)


def test_each_slice_of_a_utility_is_forbidden_where_the_utility_is():
    problem = _heating_problem()
    problem.update_options({"HENS_FORBIDDEN_MATCHES": [["HPS", "C1"]]})
    arrays = problem_to_solver_arrays(problem, D_TMIN)
    hps = next(n for n in arrays.utility_identities["hot_utilities"] if "HPS" in n)

    sliced = with_utility_slices(
        arrays, cuts={"hot": {hps: [0.0, 0.5, 1.0]}, "cold": {}}
    )

    names = sliced.utility_identities["hot_utilities"]
    assert len(names) == 3
    # One row per hot utility, one column per cold stream (C1 alone).
    forbidden = {
        name: int(row[0]) for name, row in zip(names, sliced.arrays["forbidden_hu"])
    }
    assert forbidden == {name: int(name.startswith(hps)) for name in names}
