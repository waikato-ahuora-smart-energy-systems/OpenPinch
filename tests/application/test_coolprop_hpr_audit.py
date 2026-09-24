"""Real public CoolProp proof, without replacing thermodynamics or search."""

import json
from copy import deepcopy
from time import monotonic

import pytest

from OpenPinch import PinchProblem
from OpenPinch.contracts.hpr import HPRTargetingError


@pytest.mark.parametrize(
    "mode,utility,cascade",
    [
        (mode, utility, cascade)
        for mode in ("heat_pump", "refrigeration", "mvr")
        for utility in (False, True)
        for cascade in ((True,) if mode == "mvr" else (False, True))
    ],
)
def test_real_public_coolprop_target_and_budget(mode, utility, cascade, monkeypatch):
    import OpenPinch.analysis.heat_pumps.optimisation_adapter as adapter

    counts = []
    run_search = adapter.run_hpr_candidate_search

    def measured_search(**kwargs):
        candidates = run_search(**kwargs)
        counts.append(candidates.diagnostics.evaluated_count)
        return candidates

    monkeypatch.setattr(adapter, "run_hpr_candidate_search", measured_search)
    started = monotonic()
    problem = PinchProblem("heat_pump_targeting.json")
    kwargs = dict(
        load_fraction=0.25,
        condensers=1,
        evaporators=1,
        maximum_restarts=1,
        maximum_iterations=3,
        maximum_evaluations=100,
    )
    kwargs[
        "is_utility_refrigeration"
        if mode == "refrigeration"
        else "is_utility_heat_pump"
    ] = utility
    if mode == "mvr":
        kwargs["options"] = {"HPR_REFRIGERANTS": ["Water"]}
        method = problem.target.mvr_heat_pump
    else:
        kwargs.update(refrigerants=["Water"], is_cascade_cycle=cascade)
        method = getattr(problem.target, "vapour_compression_" + mode)
    target = method(**kwargs)
    assert target is not None
    record = target.hpr_details.target_simulation_record
    assert record is not None and record.nominal_useful_duty > 0.0
    copied = deepcopy(target)
    assert copied.hpr_details.model is None
    assert copied.hpr_details.target_simulation_record == record
    assert json.loads(problem.results.model_dump_json())
    assert json.loads(record.model_dump_json())["simulation_backend"] == "coolprop"
    assert counts and all(0 < count <= 100 for count in counts)
    assert monotonic() - started < 120.0  # generous smoke gate, not a runtime SLA


def test_real_invalid_mvr_fluid_is_atomic_and_fails_before_search(monkeypatch):
    import OpenPinch.analysis.heat_pumps.optimisation_adapter as adapter

    def forbidden(**kwargs):
        pytest.fail("invalid MVR fluid entered optimization")

    monkeypatch.setattr(adapter, "run_hpr_candidate_search", forbidden)
    problem = PinchProblem("heat_pump_targeting.json")
    problem.target.direct_heat_integration()
    before = problem.results.model_dump_json()
    with pytest.raises(HPRTargetingError) as caught:
        problem.target.mvr_heat_pump(
            mvr_fluids=["not-a-fluid"],
            load_fraction=0.25,
            maximum_iterations=1,
            maximum_evaluations=1,
        )
    assert caught.value.diagnostics.evaluated_count == 0
    assert problem.results.model_dump_json() == before


@pytest.mark.parametrize("topology", ["cascade", "parallel", "mvr"])
def test_real_public_multistage_records(topology):
    problem = PinchProblem("heat_pump_targeting.json")
    kwargs = dict(
        load_fraction=0.25,
        condensers=2,
        evaporators=1,
        maximum_restarts=1,
        maximum_iterations=3,
        maximum_evaluations=150,
    )
    if topology == "mvr":
        target = problem.target.mvr_heat_pump(
            **kwargs, options={"HPR_REFRIGERANTS": ["Water"]}
        )
    else:
        target = problem.target.vapour_compression_heat_pump(
            **kwargs, refrigerants=["Water"], is_cascade_cycle=topology == "cascade"
        )
    assert target is not None
    record = target.hpr_details.target_simulation_record
    assert len(record.loops) == (3 if topology == "mvr" else 2)
    assert record.nominal_useful_duty > 0.0
    assert record.nominal_useful_duty == pytest.approx(
        sum(loop.nominal_duty for loop in record.loops)
    )
