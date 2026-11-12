"""Sparsity of the post-optimisation alpha flow-on system."""

from __future__ import annotations

import numpy as np
import pytest

from OpenPinch.analysis.heat_exchanger_networks.models._base import (
    alpha as base_alpha,
)
from tests.analysis.heat_exchanger_networks.test_base_model_helpers import (
    _alpha_state,
)


def test_alpha_system_stores_no_zero_coefficients(monkeypatch):
    # On large networks most matches are inactive, so most flow-on
    # coefficients are zero. Stored zeros fill in during the sparse LU
    # factorisation: the 64-stream pulp mill's system ran for over 20 minutes.
    model = _alpha_state([200.0, 150.0, 100.0], [180.0, 120.0, 50.0])
    # The second stage's match is off, so its flow-on terms are zero.
    model.z[0][0][1] = [0.0]
    model.Q_r[0][0][1] = [0.0]
    solve = base_alpha.spla.spsolve
    matrices = []

    def recording_solve(matrix, right_hand_side):
        matrices.append(matrix)
        return solve(matrix, right_hand_side)

    monkeypatch.setattr(base_alpha.spla, "spsolve", recording_solve)

    alpha = model.get_alpha_values()

    (matrix,) = matrices
    assert matrix.nnz == np.count_nonzero(matrix.data)
    # With no flow-on from the second stage, the first match keeps alpha 1.
    assert alpha[0][0][0][0] == pytest.approx(1.0, abs=1e-6)
