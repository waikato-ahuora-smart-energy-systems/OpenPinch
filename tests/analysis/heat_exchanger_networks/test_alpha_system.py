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


class _CountingStage(list):
    reads = 0

    def __getitem__(self, index):
        type(self).reads += 1
        return super().__getitem__(index)


def test_alpha_system_reads_each_flow_on_coefficient_once():
    # A gamma row's flow-on terms depend on the stream and stage only, not on
    # its partner; building them per partner made assembly grow with the
    # square of the stream count (seconds per pulp-mill post-process).
    hot, cold, stages = 2, 3, 3
    _CountingStage.reads = 0

    def grid(rows, columns, value):
        return [
            [_CountingStage(value for _k in range(stages)) for _c in range(columns)]
            for _r in range(rows)
        ]

    model = type("Solved", (), {})()
    model.I, model.J, model.S = hot, cold, stages
    model.beta_h = grid(hot, cold, 0.5)
    model.P_h = grid(hot, cold, 0.2)
    model.beta_c = grid(hot, cold, 0.25)
    model.P_c = grid(hot, cold, 0.4)
    model.z_j = [[1.0] * stages for _i in range(hot)]
    model.z_i = [[1.0] * stages for _j in range(cold)]

    solution = base_alpha._solve_flow_on_system(model)

    assert solution is not None
    # beta and P are each read once per (stream pair, stage that flows on).
    assert _CountingStage.reads <= 4 * hot * cold * (stages - 1)


def test_flow_on_terms_keep_only_nonzero_coefficients_in_order():
    # Most matches on a large network are inactive; dropping their zero terms
    # once per stream and stage keeps assembly from visiting them per partner.
    terms = [(0, 0.0), (1, -0.25), (2, 0.0), (3, 0.5)]

    assert base_alpha._nonzero_terms(iter(terms)) == [(1, -0.25), (3, 0.5)]


def test_flow_fractions_read_each_solved_value_once(monkeypatch):
    # Flow fractions read the same solved temperatures and duties for every
    # cell they appear in; on the pulp mill that was seconds per design.
    from OpenPinch.analysis.heat_exchanger_networks.models._base import (
        execution as base_execution,
    )
    from tests.analysis.heat_exchanger_networks.test_base_model_helpers import (
        _model,
    )

    hot, cold, stages = 2, 3, 3
    model = _model()
    model.I, model.J, model.S = hot, cold, stages
    model.non_isothermal_model = False
    model.T_h = [
        [[300.0 - 20.0 * k - i] for k in range(stages + 1)] for i in range(hot)
    ]
    model.T_c = [
        [[250.0 - 25.0 * k - j] for k in range(stages + 1)] for j in range(cold)
    ]
    model.z = [
        [[[1.0] for _k in range(stages)] for _j in range(cold)] for _i in range(hot)
    ]
    model.Q_r = [
        [[[5.0 + i + j + k] for k in range(stages)] for j in range(cold)]
        for i in range(hot)
    ]
    reads = []
    value = base_execution._value
    monkeypatch.setattr(
        base_execution, "_value", lambda item: reads.append(1) or value(item)
    )

    base_alpha._set_postoptimisation_flow_fractions(model)

    solved_values = (hot + cold) * (stages + 1) + 2 * hot * cold * stages
    assert len(reads) <= solved_values
    assert model.P_h[0][0][1] == pytest.approx((280.0 - 260.0) / (280.0 - 200.0))
