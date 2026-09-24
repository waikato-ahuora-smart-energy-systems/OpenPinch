"""Candidate correctness and public-result detachment regressions."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from hypothesis import given, seed
from hypothesis import strategies as st

from OpenPinch.analysis.heat_pumps.optimisation_adapter import (
    build_hpr_accounting,
    evaluate_hpr_candidate,
    normalise_hpr_penalty_terms,
    translate_hpr_output,
)
from OpenPinch.analysis.heat_pumps.performance_maps.target_records import (
    build_coolprop_target_simulation_record,
)
from OpenPinch.contracts.hpr import (
    HPRBackendResult,
    HPREvaluationMode,
    HPRFailureCategory,
    HPRFailureDiagnostic,
    HPRFailureSummary,
    HPRParsedState,
    HPRSearchBudget,
    HPRThermoArtifacts,
    HPRTopologyIdentifier,
)
from OpenPinch.domain.stream_collection import StreamCollection

from .helpers import _base_args


class _UncopyableEngine:
    def __deepcopy__(self, memo):
        raise TypeError("engine cannot be copied")


def _result(
    objective: float = 1.0,
    *,
    model=None,
    period_outputs=None,
) -> HPRBackendResult:
    return HPRBackendResult(
        obj=objective,
        utility_tot=objective,
        w_net=objective,
        Q_ext_heat=0.0,
        Q_ext_cold=0.0,
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
        amb_streams=StreamCollection(),
        artifacts=HPRThermoArtifacts(
            hpr_streams=StreamCollection(),
            model=model,
            debug_figure=object() if model is not None else None,
        ),
        period_outputs=period_outputs,
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2.0, (2.0,)),
        ([], ()),
        ([2.0], (2.0,)),
        ([[1.0, 2.0], [3.0, 4.0]], (1.0, 2.0, 3.0, 4.0)),
        (np.array([[1.0], [2.0]]), (1.0, 2.0)),
    ],
)
def test_penalty_terms_normalize_to_stable_finite_tuple(value, expected) -> None:
    assert normalise_hpr_penalty_terms(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        True,
        [1.0, True],
        [[1.0], [2.0, 3.0]],
        [1.0, np.nan],
        [1.0, np.inf],
        ["bad"],
    ],
)
def test_penalty_terms_reject_invalid_contract_values(value) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalise_hpr_penalty_terms(value)


@seed(20260924)
@given(
    st.lists(
        st.lists(
            st.floats(
                min_value=-1e3,
                max_value=1e3,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=0,
            max_size=5,
        ),
        min_size=0,
        max_size=5,
    ).filter(lambda rows: not rows or len({len(row) for row in rows}) <= 1)
)
def test_penalty_normalization_is_idempotent_for_rectangular_values(rows) -> None:
    normalized = normalise_hpr_penalty_terms(rows)

    assert isinstance(normalized, tuple)
    assert all(isinstance(value, float) and np.isfinite(value) for value in normalized)
    assert normalise_hpr_penalty_terms(normalized) == normalized


@seed(20260924)
@given(
    st.lists(
        st.floats(
            min_value=-1e3,
            max_value=1e3,
            allow_nan=False,
            allow_infinity=False,
        ),
        max_size=20,
    )
)
def test_penalty_accounting_matches_positive_scalar_oracle(values) -> None:
    _, _, penalty, _ = build_hpr_accounting(
        work=10.0,
        Q_ext_heat=0.0,
        Q_ext_cold=0.0,
        args=_base_args(eta_penalty=0.0, rho_penalty=2.0),
        penalty_terms=values,
    )

    assert penalty == pytest.approx(2.0 * sum(max(value, 0.0) ** 2 for value in values))
    if not values:
        assert penalty == 0.0


def test_shared_accounting_accepts_rectangular_penalty_terms() -> None:
    _, _, penalty, objective = build_hpr_accounting(
        work=10.0,
        Q_ext_heat=0.0,
        Q_ext_cold=0.0,
        args=_base_args(eta_penalty=0.0, rho_penalty=2.0),
        penalty_terms=np.array([[1.0, 2.0], [3.0, 4.0]]),
    )

    assert penalty == pytest.approx(2.0 * sum(value**2 for value in (1, 2, 3, 4)))
    assert np.isfinite(objective)


def test_search_evaluation_drops_transient_public_artifacts() -> None:
    result = evaluate_hpr_candidate(
        objective=lambda *_args, **_kwargs: _result(model=_UncopyableEngine()),
        point=[0.5],
        args=_base_args(),
        artifact_mode=HPREvaluationMode.SEARCH,
    )

    assert result.success is True
    assert result.artifacts is None
    assert result.target_simulation_record is None


@seed(20260924)
@given(
    st.lists(
        st.floats(
            min_value=-1e3,
            max_value=1e3,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=1,
        max_size=8,
    )
)
def test_search_and_final_modes_preserve_core_numerical_facts(point) -> None:
    def objective(x, _args, **_kwargs):
        return _result(objective=float(np.square(x).sum()), model=_UncopyableEngine())

    search = evaluate_hpr_candidate(
        objective=objective,
        point=point,
        args=_base_args(),
        artifact_mode=HPREvaluationMode.SEARCH,
    )
    final = evaluate_hpr_candidate(
        objective=objective,
        point=point,
        args=_base_args(),
        artifact_mode=HPREvaluationMode.FINAL,
    )

    assert np.isfinite(search.obj)
    assert search.obj == final.obj
    assert search.utility_tot == final.utility_tot
    assert search.w_net == final.w_net
    assert search.artifacts is None
    assert final.artifacts is not None


@pytest.mark.parametrize(
    ("topology", "n_vc", "n_mvr", "expected_roles"),
    [
        (
            HPRTopologyIdentifier.CASCADE_VAPOUR_COMPRESSION,
            2,
            0,
            ("vapour_compression",) * 2,
        ),
        (
            HPRTopologyIdentifier.PARALLEL_VAPOUR_COMPRESSION,
            2,
            0,
            ("vapour_compression",) * 2,
        ),
        (
            HPRTopologyIdentifier.VAPOUR_COMPRESSION_MVR,
            1,
            1,
            ("vapour_compression", "mvr"),
        ),
    ],
)
def test_generalized_coolprop_record_preserves_ordered_detached_loops(
    topology: HPRTopologyIdentifier,
    n_vc: int,
    n_mvr: int,
    expected_roles: tuple[str, ...],
) -> None:
    args = _base_args(
        n_cond=n_vc,
        n_evap=n_vc,
        n_mvr=n_mvr,
        refrigerant_ls=["R134A"] * n_vc,
        mvr_fluid_ls=["Water"] * n_mvr,
        eta_mvr_comp=0.7,
        eta_motor=0.95,
    )
    count = n_vc + n_mvr
    state = HPRParsedState(
        T_evap=np.arange(count, dtype=float) * 10.0 + 40.0,
        T_cond=np.arange(count, dtype=float) * 10.0 + 60.0,
        dT_subcool=np.ones(count),
        dT_ihx_gas_side=np.zeros(count),
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
    )
    cycle = type(
        "DetachedFakeCycle",
        (),
        {
            "Q_heat_arr": np.arange(count, dtype=float) * 10.0 + 100.0,
            "Q_cool_arr": np.arange(count, dtype=float) * 10.0 + 80.0,
            "work_arr": np.arange(count, dtype=float) + 20.0,
            "dT_superheat": np.full(count, 5.0),
        },
    )()

    record = build_coolprop_target_simulation_record(
        args=args,
        state=state,
        cycle=cycle,
        topology_id=topology,
    )

    assert record is not None
    assert record.topology_id is topology
    assert tuple(loop.loop_role for loop in record.loops) == expected_roles
    assert [loop.ordinal for loop in record.loops] == list(range(count))
    assert record.model_dump(mode="json")["loops"]


def test_final_translation_omits_uncopyable_engine_and_is_deepcopy_safe() -> None:
    output = translate_hpr_output(_result(model=_UncopyableEngine()))

    assert output.model is None
    assert deepcopy(output) == output


def test_nested_period_outputs_are_detached_from_engine_models() -> None:
    period = _result(model=_UncopyableEngine())
    output = translate_hpr_output(
        _result(period_outputs={"winter": period}),
    )

    detached_period = output.period_outputs["winter"]
    assert isinstance(detached_period, dict)
    assert "model" not in detached_period
    deepcopy(output)


@seed(20260924)
@given(
    st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8),
        min_size=1,
        max_size=8,
        unique=True,
    )
)
def test_recursive_period_finalization_is_ordered_detached_and_copy_safe(
    period_ids: list[str],
) -> None:
    output = translate_hpr_output(
        _result(
            period_outputs={
                period_id: _result(model=_UncopyableEngine())
                for period_id in period_ids
            }
        )
    )

    assert list(output.period_outputs) == period_ids
    assert all(
        isinstance(period, dict) and "model" not in period and "artifacts" not in period
        for period in output.period_outputs.values()
    )
    assert deepcopy(output) == output


@seed(20260924)
@given(
    maximum_iterations=st.integers(min_value=1, max_value=10_000),
    maximum_evaluations=st.integers(min_value=1, max_value=1_000_000),
    representative_count=st.integers(min_value=0, max_value=20),
)
def test_reliability_contract_round_trip_and_diagnostic_cap(
    maximum_iterations: int,
    maximum_evaluations: int,
    representative_count: int,
) -> None:
    budget = HPRSearchBudget(
        maximum_iterations=maximum_iterations,
        maximum_evaluations=maximum_evaluations,
    )
    diagnostic = HPRFailureDiagnostic(
        category=HPRFailureCategory.CANDIDATE_PHYSICAL_INFEASIBILITY,
        reason_code="coolprop.flash_infeasible",
        summary="candidate is outside the fluid envelope",
    )
    values = {
        "simulation_backend": "coolprop",
        "cycle": "cascade_vapour_compression",
        "evaluated_count": representative_count,
        "category_counts": {
            HPRFailureCategory.CANDIDATE_PHYSICAL_INFEASIBILITY: representative_count
        },
        "representative_failures": (diagnostic,) * representative_count,
        "budget": budget,
        "warm_start_evaluated": True,
        "warm_start_viable": False,
    }
    if representative_count > 16:
        with pytest.raises(ValueError):
            HPRFailureSummary(**values)
        return

    summary = HPRFailureSummary(**values)
    assert HPRFailureSummary.model_validate_json(summary.model_dump_json()) == summary
    assert HPRSearchBudget.model_validate_json(budget.model_dump_json()) == budget


@seed(20260924)
@given(
    topology=st.sampled_from(
        [
            HPRTopologyIdentifier.CASCADE_VAPOUR_COMPRESSION,
            HPRTopologyIdentifier.PARALLEL_VAPOUR_COMPRESSION,
        ]
    ),
    count=st.integers(min_value=1, max_value=5),
)
def test_generalized_coolprop_record_order_property(topology, count) -> None:
    args = _base_args(
        n_cond=count,
        n_evap=count,
        refrigerant_ls=["R134A"] * count,
    )
    state = HPRParsedState(
        T_evap=np.arange(count, dtype=float) + 40.0,
        T_cond=np.arange(count, dtype=float) + 60.0,
        dT_subcool=np.ones(count),
        dT_ihx_gas_side=np.zeros(count),
    )
    cycle = type(
        "DetachedPropertyCycle",
        (),
        {
            "Q_heat_arr": np.arange(count, dtype=float) + 100.0,
            "Q_cool_arr": np.arange(count, dtype=float) + 80.0,
            "work_arr": np.arange(count, dtype=float) + 20.0,
        },
    )()

    record = build_coolprop_target_simulation_record(
        args=args,
        state=state,
        cycle=cycle,
        topology_id=topology,
    )

    assert record.topology_id is topology
    assert [loop.ordinal for loop in record.loops] == list(range(count))
    assert len({loop.loop_id for loop in record.loops}) == count
    assert all(np.isfinite(loop.nominal_duty) for loop in record.loops)
