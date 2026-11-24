"""The interior cuts read a non-isothermal match on its own branch.

In the non-isothermal stage-wise model a split stream's branches leave a
stage at their own temperatures, and the stage's mixed outlet lies
between them. The interior approach check of a match, and its cut, use
the match's branch outlets (``T_h_out_x``, ``T_c_out_y``), not the stage's.
"""

from __future__ import annotations

from types import SimpleNamespace

from OpenPinch.analysis.heat_exchanger_networks.models._base import (
    interior_approach,
)


def test_a_non_isothermal_match_is_checked_on_its_own_branch():
    owner = SimpleNamespace(
        non_isothermal_model=True,
        T_h_by_period=[[[200.0, 170.0]]],
        T_h_mid_by_period=[[[200.0, 170.0]]],
        T_h_out_x_by_period=[[[[165.0]]]],
        T_c_by_period=[[[150.0, 120.0]]],
        T_c_mid_by_period=[[[150.0, 120.0]]],
        T_c_out_y_by_period=[[[[155.0]]]],
    )

    ends = interior_approach._branch_ends(owner, 0, 0, 0, 0)

    assert ends == (200.0, 165.0, 120.0, 155.0)
