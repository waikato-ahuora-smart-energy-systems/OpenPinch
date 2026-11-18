"""Sliding-temperature utilities sliced at the streams they must reach.

A heater runs its utility over the utility's whole range, so hot water
(90 to 70 C) cannot heat a stream entering above 85 C with a 5 K approach.
Slicing the utility at each stream's supply plus approach, with the slices
in series, lets a stream use the slices it can reach.
"""

from __future__ import annotations

import numpy as np
import pytest

from OpenPinch.analysis.heat_exchanger_networks.solver import (
    pinch_design_decomposition as decomposition_module,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.utility_slices import (
    sliced_decomposition,
    with_utility_slices,
)
from tests.analysis.heat_exchanger_networks.test_pdm_level_targets import _process

D_TMIN = 10.0
HW = "Hot Utility.HW"


@pytest.fixture(scope="module")
def hospital():
    return _process("Hospital[S]")


@pytest.fixture(scope="module")
def arrays(hospital):
    return problem_to_solver_arrays(hospital, D_TMIN)


def _hot_axis(arrays) -> list[str]:
    return list(arrays.utility_identities["hot_utilities"])


def test_hot_water_splits_where_cooking_and_sterilisation_need_it(arrays):
    sliced = with_utility_slices(arrays)

    slices = sliced.preparation["utility_slices"]["hot_utilities"]
    hot_water = [name for name, (parent, _f) in slices.items() if parent == HW]
    index = [sliced.axis_maps["hot_utilities"][name] for name in hot_water]
    inlets = sliced.arrays["T_hu_in_period"][0][index] - 273.15
    outlets = sliced.arrays["T_hu_out_period"][0][index] - 273.15
    # Sterilisation enters at 82 C and Cooking at 80 C, each 5.1 K short.
    np.testing.assert_allclose(inlets, [90.0, 87.1, 85.1], atol=1e-9)
    np.testing.assert_allclose(outlets, [87.1, 85.1, 70.0], atol=1e-9)
    fractions = [slices[name][1] for name in hot_water]
    np.testing.assert_allclose(fractions, [0.145, 0.1, 0.755], atol=1e-9)


def test_steam_and_every_other_utility_property_is_kept(arrays):
    sliced = with_utility_slices(arrays)

    steam = [name for name in _hot_axis(arrays) if name != HW]
    for name in steam:
        old = arrays.axis_maps["hot_utilities"][name]
        new = sliced.axis_maps["hot_utilities"][name]
        for key in ("T_hu_in_period", "T_hu_out_period", "hu_cost_period"):
            assert sliced.arrays[key][0][new] == arrays.arrays[key][0][old]
    parent = arrays.axis_maps["hot_utilities"][HW]
    for name in _hot_axis(sliced):
        if name.startswith(HW + "@"):
            new = sliced.axis_maps["hot_utilities"][name]
            for key in ("T_hu_cont_period", "htc_hu_period", "hu_cost_period"):
                assert sliced.arrays[key][0][new] == arrays.arrays[key][0][parent]


def test_a_problem_with_no_cut_is_returned_as_it_is():
    from tests.analysis.heat_exchanger_networks.test_pdm_staged_levels import (
        HEATING,
        _problem,
    )

    arrays = problem_to_solver_arrays(_problem(HEATING), D_TMIN)

    assert with_utility_slices(arrays) is arrays


def test_each_slice_takes_its_share_of_the_levels_target(hospital, arrays):
    sliced = with_utility_slices(arrays)
    decomposition = decomposition_module.build_pinch_design_decomposition(
        hospital, D_TMIN, pinch_location="above"
    )

    result = sliced_decomposition(decomposition, sliced)

    assert result.hot_utility_identities == tuple(_hot_axis(sliced))
    before = dict(
        zip(
            decomposition.hot_utility_identities,
            decomposition.period_targets[0].hot_utility_level_targets,
        )
    )
    after = dict(
        zip(
            result.hot_utility_identities,
            result.period_targets[0].hot_utility_level_targets,
        )
    )
    slices = sliced.preparation["utility_slices"]["hot_utilities"]
    for name, duty in after.items():
        parent, fraction = slices.get(name, (name, 1.0))
        assert duty == pytest.approx(before[parent] * fraction, rel=1e-12)
    assert sum(after.values()) == pytest.approx(sum(before.values()), rel=1e-12)


def _pdm(problem, **kwargs):
    from OpenPinch.analysis.heat_exchanger_networks.models.problem import (
        InternalHeatExchangerNetworkProblem,
    )

    return InternalHeatExchangerNetworkProblem(
        solver_arrays=problem_to_solver_arrays(problem, D_TMIN),
        name="sliced-pdm",
        framework="PDM",
        solver="highs",
        dTmin=D_TMIN,
        minimisation_goal="total utility",
        pinch_decompositions={
            location: decomposition_module.build_pinch_design_decomposition(
                problem, D_TMIN, pinch_location=location
            )
            for location in ("above", "below")
        },
        staged_utilities=True,
        **kwargs,
    )


def _utility_duties(network) -> dict[str, float]:
    from OpenPinch.domain.enums import HeatExchangerKind

    duties: dict[str, float] = {}
    for exchanger in network.exchangers:
        if exchanger.kind is HeatExchangerKind.RECOVERY:
            continue
        utility = (
            exchanger.source_stream
            if exchanger.kind is HeatExchangerKind.HOT_UTILITY
            else exchanger.sink_stream
        )
        duties[utility] = duties.get(utility, 0.0) + exchanger.state().duty
    return duties


@pytest.mark.parametrize("heuristic_network_limit", [0, 3], ids=["milp", "heuristic"])
def test_hospital_holds_hot_water_at_its_target_through_slices(
    hospital, heuristic_network_limit, caplog
):
    from OpenPinch.analysis.heat_exchanger_networks.reporting.verification import (
        verify_network_feasibility,
    )

    decomposition = decomposition_module.build_pinch_design_decomposition(
        hospital, D_TMIN, pinch_location="above"
    )
    pdm = _pdm(hospital, heuristic_network_limit=heuristic_network_limit)

    with caplog.at_level("WARNING"):
        case = pdm.get_solution(print_output=False)

    assert case is not None, getattr(pdm, "solution_failure_reason", None)
    assert "utility levels" not in caplog.text
    network = pdm.extract_network(run_id="sliced-pdm")
    duties = _utility_duties(network)
    targets = dict(
        zip(
            decomposition.hot_utility_identities,
            decomposition.period_targets[0].hot_utility_level_targets,
        )
    )
    # Slices are reported under their parent utility.
    assert duties.get(HW, 0.0) == pytest.approx(targets[HW], rel=1e-6)
    assert all("@" not in name for name in duties)
    assert len({e.exchanger_id for e in network.exchangers}) == len(network.exchangers)
    assert verify_network_feasibility(network) == ()


def _slice_duties(model, sliced) -> dict[str, float]:
    """Each hot utility slice's duty in period 0, stream-end and staged."""
    from OpenPinch.analysis.heat_exchanger_networks.models._base.execution import (
        _value,
    )

    names = list(sliced.utility_identities["hot_utilities"])
    duties = dict.fromkeys(names, 0.0)
    for row in model.Q_h_by_period[0]:
        for u, duty in enumerate(row):
            duties[names[u]] += _value(duty)
    for row in getattr(model, "Q_hs_by_period", [[]])[0]:
        for u, by_k in enumerate(row):
            duties[names[u]] += sum(_value(duty) for duty in by_k)
    return duties


def test_slices_run_in_series_when_the_levels_are_not_held(hospital, arrays):
    """Priced freely, a slice still takes only its share of its utility."""
    from dataclasses import replace

    from OpenPinch.analysis.heat_exchanger_networks.models.pinch_decomposition import (
        PinchDecompModel,
    )

    sliced = with_utility_slices(arrays)
    # Hot water the cheapest, so the side uses its slices.
    costs = sliced.arrays["hu_cost_period"].copy()
    for name, index in sliced.axis_maps["hot_utilities"].items():
        if name.startswith(HW + "@"):
            costs[:, index] = 1.0
    sliced = replace(sliced, arrays=sliced.arrays | {"hu_cost_period": costs})
    model = PinchDecompModel(
        name="above-sliced",
        framework="PDM",
        solver="highs",
        solver_arrays=sliced,
        dTmin=D_TMIN,
        z_restriction=None,
        min_dqda=0,
        minimisation_goal="hot utility",
        non_isothermal_model=False,
        integers=True,
        tol=1e-3,
        pinch_loc="above",
        pinch_decomposition=sliced_decomposition(
            decomposition_module.build_pinch_design_decomposition(
                hospital, D_TMIN, pinch_location="above"
            ),
            sliced,
        ),
        stage_selection="automated",
    )
    model.optimise(False)
    assert model.mSuccess == 1

    duties = _slice_duties(model, sliced)
    slices = sliced.preparation["utility_slices"]["hot_utilities"]
    total = sum(duties[name] for name in slices)
    assert total > 1.0
    for name, (_parent, share) in slices.items():
        assert duties[name] == pytest.approx(total * share, abs=1e-6)


def _couenne_available() -> bool:
    from OpenPinch.analysis.heat_exchanger_networks.solver import backend
    from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
        MissingSynthesisDependencyError,
        MissingSynthesisSolverError,
    )

    try:
        backend.require_solver_backend("couenne")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


@pytest.mark.solver
@pytest.mark.skipif("not _couenne_available()", reason="needs couenne")
def test_a_stage_wise_task_takes_on_its_parents_slices(hospital):
    from OpenPinch.analysis.heat_exchanger_networks.models.problem import (
        InternalHeatExchangerNetworkProblem,
    )

    pdm = _pdm(hospital)
    assert pdm.get_solution(print_output=False) is not None
    tdm = InternalHeatExchangerNetworkProblem(
        solver_arrays=problem_to_solver_arrays(hospital, D_TMIN),
        name="sliced-tdm",
        framework="TDM",
        solver="couenne",
        dTmin=D_TMIN,
        minimisation_goal="total cost",
        parent=pdm,
    )

    tdm.load_model()

    assert tdm.solver_arrays.utility_identities == pdm.solver_arrays.utility_identities
    assert tdm.case.NHU == pdm.case.NHU
