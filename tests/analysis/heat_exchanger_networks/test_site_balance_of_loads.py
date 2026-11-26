"""The site balance of fixed loads needs no synthesis solver.

A site design that falls back to its joined process networks prices them
with this balance. The design itself may have run on SCIP and Ipopt alone
(the ``highs`` extra is optional), so the balance, a small linear
programme, is solved with SciPy.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.site.utility_system import (
    balance_of_loads,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisSolverError,
)
from OpenPinch.contracts.synthesis.site import SiteUtilityLevel, SiteUtilityLevels


def _levels() -> SiteUtilityLevels:
    """A boiler (VHPS), one steam main (LPS) and cooling water (CW)."""
    return SiteUtilityLevels(
        levels=(
            SiteUtilityLevel(
                name="VHPS",
                kind="hot",
                t_high=(250.1,),
                t_low=(250.0,),
                import_price=(40.0,),
            ),
            SiteUtilityLevel(
                name="LPS",
                kind="both",
                t_high=(150.1,),
                t_low=(150.0,),
                import_price=(30.0,),
                dump_price=(30.0,),
            ),
            SiteUtilityLevel(
                name="CW",
                kind="cold",
                t_high=(25.0,),
                t_low=(20.0,),
                dump_price=(5.0,),
            ),
        )
    )


def test_the_balance_of_fixed_loads_is_solved_without_a_synthesis_solver(monkeypatch):
    def no_solver(*arguments, **options):
        raise MissingSynthesisSolverError("highs is not installed")

    monkeypatch.setattr(backend, "solve_pyomo_model", no_solver)

    balance = balance_of_loads(_levels(), {"LPS": (10.0,)}, {"LPS": (25.0,)})

    # The 15 kW surplus goes down to cooling water at 5 $, not out at 30 $.
    assert balance.level("LPS").letdown_out == pytest.approx(15.0, abs=1e-7)
    assert balance.level("CW").dump == pytest.approx(15.0, abs=1e-7)
    assert balance.operating_cost == pytest.approx(75.0, abs=1e-5)
