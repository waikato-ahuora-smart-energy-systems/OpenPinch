"""Distinct alternative designs close in cost to the selected one."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

import OpenPinch.analysis.heat_pumps.optimisation_adapter as adapter
from OpenPinch.contracts.hpr import (
    HPRActiveStage,
    HPRBackendResult,
    HPREffectiveTopology,
    HPREvaluationMode,
    HPRThermoArtifacts,
)
from OpenPinch.domain.enums import HeatPumpAndRefrigerationCycle
from OpenPinch.domain.stream_collection import StreamCollection
from OpenPinch.optimisation.errors import NoOptimisationCandidatesError

from .helpers import _base_args

# Point -> (objective, work per stage). Two stages; the signature is which
# stages work (and their fluids).
_DESIGNS = {
    0.10: (100.0, [10.0, 0.0]),  # stages {0}: the selected design
    0.15: (101.0, [9.0, 0.0]),  # stages {0} again, so not an alternative
    0.20: (102.0, [0.0, 10.0]),  # stages {1}: 2% dearer
    0.30: (103.0, [10.0, 10.0]),  # stages {0, 1}: 3% dearer
    0.40: (110.0, [5.0, 5.0, 5.0]),  # 10% dearer: outside the window
}


def _objective(x, args, *, debug=False, artifact_mode=HPREvaluationMode.FINAL):
    obj, works = _DESIGNS[round(float(x[0]), 2)]
    result = HPRBackendResult(
        obj=obj,
        utility_tot=obj,
        w_net=sum(works),
        w_hpr=np.array(works),
        Q_ext_heat=0.0,
        Q_ext_cold=0.0,
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
    )
    if artifact_mode is HPREvaluationMode.FINAL:
        active = tuple(
            HPRActiveStage(
                index=i,
                role="vapour_compression",
                fluid="R134A",
                evaporating_temperature=10.0,
                condensing_temperature=60.0,
                work=w,
            )
            for i, w in enumerate(works)
            if w > 0
        )
        result = result.with_updates(
            artifacts=HPRThermoArtifacts(hpr_streams=StreamCollection()),
            effective_topology=HPREffectiveTopology.from_stages(
                cycle="cascade",
                condensers=1,
                evaporators=2,
                configured_stages=len(works),
                active_stages=active,
            ),
        )
    return result


def _optimiser(problem, *, method, options):
    for x in _DESIGNS:
        problem.objective(np.array([x]), *problem.args)
    raise NoOptimisationCandidatesError("none")


def _solve():
    return adapter.solve_hpr_placement(
        f_obj=_objective,
        x0_ls=None,
        bnds=[(0.0, 1.0)],
        args=_base_args(
            hpr_type=HeatPumpAndRefrigerationCycle.CascadeVapourComp.value,
            max_multi_start=1,
        ),
        candidate_search=partial(
            adapter.run_hpr_candidate_search, optimiser=_optimiser
        ),
    )


def test_distinct_designs_within_five_percent_are_reported_cheapest_first():
    result = _solve()

    assert result.obj == pytest.approx(100.0)
    alternatives = result.alternatives
    assert [a.objective for a in alternatives] == pytest.approx([102.0, 103.0])
    assert [a.objective_gap for a in alternatives] == pytest.approx([0.02, 0.03])
    assert [
        [stage.index for stage in a.topology.active_stages] for a in alternatives
    ] == [[1], [0, 1]]


def test_the_alternatives_reach_the_public_output():
    output = adapter.translate_hpr_output(_solve())

    assert len(output.alternatives) == 2


def test_an_acceptable_point_is_a_candidate_when_penalised_points_rank_first():
    # Seventeen slightly penalised points rank below the one feasible design
    # (cost plus penalty 50 against 100), crowding it out of the 16 kept
    # viable points. The feasible design is still the target.
    feasible = 0.9
    points = [round(0.01 * i, 2) for i in range(17)] + [feasible]

    def objective(x, args, *, debug=False, artifact_mode=HPREvaluationMode.FINAL):
        works = [10.0, 0.0]
        penalised = round(float(x[0]), 2) != feasible
        result = HPRBackendResult(
            obj=50.0 if penalised else 100.0,
            utility_tot=1.0,
            w_net=10.0,
            w_hpr=np.array(works),
            Q_ext_heat=0.0,
            Q_ext_cold=0.0,
            Q_amb_hot=0.0,
            Q_amb_cold=0.0,
            feasibility_penalty=5.0 if penalised else 0.0,
        )
        if artifact_mode is HPREvaluationMode.FINAL:
            result = result.with_updates(
                artifacts=HPRThermoArtifacts(hpr_streams=StreamCollection())
            )
        return result

    def optimiser(problem, *, method, options):
        for x in points:
            problem.objective(np.array([x]), *problem.args)
        raise NoOptimisationCandidatesError("none")

    result = adapter.solve_hpr_placement(
        f_obj=objective,
        x0_ls=None,
        bnds=[(0.0, 1.0)],
        args=_base_args(
            hpr_type=HeatPumpAndRefrigerationCycle.CascadeVapourComp.value,
            max_multi_start=1,
        ),
        candidate_search=partial(adapter.run_hpr_candidate_search, optimiser=optimiser),
    )

    assert result.obj == pytest.approx(100.0)
