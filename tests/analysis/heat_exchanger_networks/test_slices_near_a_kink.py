"""An exchanger whose inlet lies just past a segment kink is sliced.

The fixed-structure NLP rounds each kink over 0.05 K, so a hot inlet 0.09 K
past a kink holds a little less heat in the solver than on the exact profile
the slices are cut on. The duty then overshot the exact profile by 0.05 kW,
and ``optimise_duties`` with the ``cost`` or ``area`` objective raised
"Exchanger duty exceeds an available parent segment profile" on two of three
pinch design networks of a segmented case (review finding 27).
"""

from __future__ import annotations

import numpy as np
import pytest

from OpenPinch.analysis.heat_exchanger_networks.models.fixed_structure import _slices
from OpenPinch.analysis.heat_exchanger_networks.solver.piecewise import (
    PiecewiseThermalProfile,
)


def _curve(identities, t_in, t_out, duties) -> PiecewiseThermalProfile:
    t_in, t_out, duties = (np.array(v, dtype=float) for v in (t_in, t_out, duties))
    return PiecewiseThermalProfile(
        identities=tuple(identities),
        temperatures_in=t_in,
        temperatures_out=t_out,
        duties=duties,
        heat_capacity_flowrates=duties / np.abs(t_in - t_out),
        heat_transfer_coefficients=np.ones(len(duties)),
    )


def test_an_exchanger_entering_just_past_a_kink_is_sliced():
    # H1 releases 2,000 kW down to 503.15 K, then 800 kW to 373.15 K; the
    # solver's inlet of 503.0633 K holds 2,000.53 kW on the exact profile.
    hot = _curve(("H1.a", "H1.b"), (653.15, 503.15), (503.15, 373.15), (2000, 800))
    duty = 799.514322816647
    cold = _curve(("C1",), (303.15,), (323.15,), (duty,))

    slices = _slices(hot, cold, duty, 503.0633, 303.15, "0")

    assert sum(s.duty for s in slices) == pytest.approx(duty, abs=0.1)
