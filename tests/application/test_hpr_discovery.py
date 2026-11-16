"""Discovering the registered cycles, and the gateway over all periods and commands.

``target.hpr_cycles()`` describes every registered cycle, with its options
model's JSON Schema, so a client builds its cycle picker and option form
from it. A cycle registered later (a test-only one here) runs through the
gateway, the command layer and discovery with no other change.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_pumps.cycles import registry
from OpenPinch.application._problem.accessors.target import _TargetAccessor
from OpenPinch.application.commands import run_command
from OpenPinch.contracts.hpr import (
    HPRFailureSummary,
    HprLevels,
    HprLoad,
    HPRSearchBudget,
    HprSearchLimits,
    HPRTargetingError,
)

# Two short periods: these run in the Pyodide lane, where the crude case's
# three periods took over a minute per test.
from tests.analysis.heat_exchanger_networks.test_hen_period_handling import (
    _two_period_problem,
)

CHEAP = dict(
    load=HprLoad(fraction=0.25),
    levels=HprLevels(condensers=1, evaporators=1),
    # A bounded search, as the notebooks use: the Pyodide lane runs these.
    search=HprSearchLimits(restarts=1, iterations=20, evaluations=50),
)


def test_hpr_cycles_describes_every_registered_cycle():
    catalog = PinchProblem("basic_pinch.json").target.hpr_cycles()

    by_id = {cycle.id: cycle for cycle in catalog.cycles}
    assert list(by_id) == ["carnot", "vapour_compression", "mvr", "brayton"]
    assert by_id["mvr"].duties == ("heat_pump",)
    assert by_id["carnot"].topologies == ("cascade", "parallel")
    assert by_id["carnot"].default_topology == "cascade"
    assert not by_id["brayton"].available and not by_id["brayton"].all_periods
    assert by_id["vapour_compression"].dependencies == (
        "tespy (explicit backend only)",
    )
    schema = by_id["vapour_compression"].options_schema
    assert {"refrigerants", "compressor_efficiency"} <= set(schema["properties"])


def test_the_cycle_catalogue_is_a_command():
    response = run_command(
        {"command": "target.hpr_cycles", "problem": {"sample_case": "basic_pinch.json"}}
    )

    assert [cycle["id"] for cycle in response["value"]["cycles"]] == [
        "carnot",
        "vapour_compression",
        "mvr",
        "brayton",
    ]


def _hpr_work(outputs) -> dict[str, float]:
    """Each period's heat pump work, from the target that carries it."""
    work = {
        period: next(
            target.hpr_work.value for target in output.targets if target.hpr_success
        )
        for period, output in outputs.items()
    }
    assert work and all(value > 0.0 for value in work.values())
    return work


def test_the_gateway_over_all_periods_matches_the_named_method():
    def work(run):
        problem = _two_period_problem()
        return _hpr_work(run(problem.target.all_periods))

    named = work(
        lambda periods: periods.carnot_heat_pump(
            load_fraction=0.25,
            condensers=1,
            evaporators=1,
            maximum_restarts=1,
            maximum_iterations=20,
            maximum_evaluations=50,
        )
    )
    gateway = work(
        lambda periods: periods.heat_pump_and_refrigeration(cycle="carnot", **CHEAP)
    )

    assert gateway == named


def test_a_gateway_command_validates_json_cycle_options(monkeypatch):
    seen = {}

    def capture(self, **kwargs):
        seen.update(kwargs)
        raise HPRTargetingError("stopped", diagnostics=_diagnostics())

    monkeypatch.setattr(_TargetAccessor, "_hpr", capture)

    response = run_command(
        {
            "command": "target.heat_pump_and_refrigeration",
            "problem": {"sample_case": "basic_pinch.json"},
            "arguments": {
                "cycle": "vapour_compression",
                "cycle_options": {
                    "refrigerants": ["Ammonia"],
                    "sort_refrigerants": True,
                },
            },
        }
    )

    assert response["error"]["type"] == "HPRTargetingError"
    assert seen["extra_configuration"] == {
        "HPR_REFRIGERANTS": ["Ammonia"],
        "HPR_REFRIGERANT_SORT_ENABLED": True,
    }


def test_a_bad_cycle_option_is_a_value_error_naming_it():
    with pytest.raises(ValueError, match="mvr_stages"):
        run_command(
            {
                "command": "target.heat_pump_and_refrigeration",
                "problem": {"sample_case": "basic_pinch.json"},
                "arguments": {"cycle": "carnot", "cycle_options": {"mvr_stages": 2}},
            }
        )


def test_a_gateway_targeting_failure_is_returned_as_an_error(monkeypatch):
    def infeasible(self, **_kwargs):
        raise HPRTargetingError("bounded failure", diagnostics=_diagnostics())

    monkeypatch.setattr(_TargetAccessor, "_hpr", infeasible)

    response = run_command(
        {
            "command": "target.heat_pump_and_refrigeration",
            "problem": {"sample_case": "basic_pinch.json"},
        }
    )

    assert response["error"]["type"] == "HPRTargetingError"
    assert response["error"]["details"]["status"] == "typed infeasible"


def test_a_gateway_target_is_a_reference_for_the_refrigerant_screen():
    session = run_command(
        {"command": "open", "problem": {"sample_case": "basic_pinch.json"}}
    )["session"]
    reference = run_command(
        {
            "command": "target.heat_pump_and_refrigeration",
            "session": session,
            "arguments": {
                "cycle": "carnot",
                "load": {"fraction": 0.25},
                "levels": {"condensers": 1, "evaporators": 1},
                "search": {"restarts": 1},
            },
        }
    )["value"]["ref"]

    screen = run_command(
        {
            "command": "target.hpr_refrigerant_screen",
            "session": session,
            "arguments": {"target": {"$ref": reference}, "refrigerants": "natural"},
        }
    )

    assert screen["value"]["source"] == "carnot_target"


class DemoCycleOptions(BaseModel):
    """Options of a test-only cycle."""

    model_config = ConfigDict(extra="forbid")
    cycle: Literal["demo"] = "demo"
    compressor_efficiency: float | None = Field(default=None, gt=0.0, le=1.0)


@pytest.fixture
def demo_cycle(monkeypatch):
    """A cycle registered in place of carnot, reusing its model families."""

    carnot = registry.cycle_spec("carnot")
    cycles = {
        name: spec for name, spec in registry._REGISTRY.items() if name != "carnot"
    }
    monkeypatch.setattr(registry, "_REGISTRY", cycles)
    registry.register_cycle(
        replace(
            carnot,
            id="demo",
            label="Demo cycle",
            duties=frozenset({"heat_pump"}),
            options_model=DemoCycleOptions,
            configuration_keys={"compressor_efficiency": "HPR_ETA_COMP"},
        )
    )
    return "demo"


def test_a_newly_registered_cycle_runs_end_to_end(demo_cycle):
    target = PinchProblem("basic_pinch.json").target.heat_pump_and_refrigeration(
        demo_cycle, cycle_options={"compressor_efficiency": 0.6}, **CHEAP
    )
    assert target.hpr_success

    response = run_command(
        {
            "command": "target.heat_pump_and_refrigeration",
            "problem": {"sample_case": "basic_pinch.json"},
            "arguments": {
                "cycle": demo_cycle,
                "load": {"fraction": 0.25},
                "levels": {"condensers": 1, "evaporators": 1},
                "search": {"restarts": 1},
                "cycle_options": {"compressor_efficiency": 0.6},
            },
        }
    )
    assert response.get("error") is None

    discovered = PinchProblem("basic_pinch.json").target.hpr_cycles()
    demo = next(cycle for cycle in discovered.cycles if cycle.id == demo_cycle)
    assert demo.label == "Demo cycle"
    assert demo.options_schema == DemoCycleOptions.model_json_schema()


def _diagnostics() -> HPRFailureSummary:
    return HPRFailureSummary(
        simulation_backend="coolprop",
        cycle="single-stage",
        evaluated_count=0,
        category_counts={},
        representative_failures=(),
        budget=HPRSearchBudget(maximum_iterations=1, maximum_evaluations=1),
        warm_start_evaluated=False,
        warm_start_viable=False,
    )


def test_all_periods_runs_a_request_once_per_period():
    from OpenPinch.contracts.hpr import HprTargetRequest

    work = _hpr_work
    named = _two_period_problem().target.all_periods
    request = HprTargetRequest(cycle="carnot", **CHEAP)
    gateway = _two_period_problem().target.all_periods

    assert work(gateway.heat_pump_and_refrigeration(request=request)) == work(
        named.carnot_heat_pump(
            load_fraction=0.25,
            condensers=1,
            evaporators=1,
            maximum_restarts=1,
            maximum_iterations=20,
            maximum_evaluations=50,
        )
    )


def test_a_request_naming_another_period_than_the_call_is_rejected():
    from OpenPinch.contracts.hpr import HprTargetRequest

    problem = _two_period_problem()

    with pytest.raises(ValueError, match="period"):
        problem.target.heat_pump_and_refrigeration(
            request=HprTargetRequest(period_id="peak"), period_id="base"
        )


def test_a_new_cycles_recorded_run_replays_as_that_cycle(demo_cycle):
    # carnot is not registered here, so a replay falling back to the default
    # cycle would fail.
    from OpenPinch.application._problem.periods import (
        aggregation as period_aggregation,
    )

    problem = PinchProblem("basic_pinch.json")
    problem.target.heat_pump_and_refrigeration(demo_cycle, **CHEAP)
    spec = problem._last_target_run_spec

    replayed = period_aggregation.target_output_for_recorded_period(problem, spec, "0")

    assert any(target.hpr_success for target in replayed.targets)


def test_a_recorded_named_run_replays_its_own_topology(monkeypatch):
    # Replaying a recorded run used to fail on the HPR_TYPE it recorded, and
    # the named method's defaults would have replayed a parallel run as a
    # cascade.
    from OpenPinch.application._problem.periods import (
        aggregation as period_aggregation,
    )

    problem = PinchProblem("basic_pinch.json")
    problem.target.carnot_heat_pump(
        is_cascade_cycle=False,
        load_fraction=0.25,
        condensers=1,
        evaporators=1,
        maximum_restarts=1,
    )
    spec = problem._last_target_run_spec
    seen = []
    real = _TargetAccessor._execute

    def capture(self, **kwargs):
        # The recorded options come back as options; the method's own
        # configuration wins over them.
        seen.append(dict(kwargs["options"] or {}) | dict(kwargs["configuration"]))
        return real(self, **kwargs)

    monkeypatch.setattr(_TargetAccessor, "_execute", capture)

    period_aggregation.target_output_for_recorded_period(problem, spec, "0")

    assert seen[-1]["HPR_TYPE"] == "Parallel Carnot cycles"
    assert seen[-1]["HPR_LOAD_FRACTION"] == 0.25


class StirlingOptions(BaseModel):
    """Options of a test-only cycle with a model family of its own."""

    model_config = ConfigDict(extra="forbid")
    cycle: Literal["stirling"] = "stirling"


def test_a_cycle_with_a_model_family_of_its_own_runs_end_to_end(monkeypatch):
    # No HeatPumpAndRefrigerationCycle member: the cycle names its family,
    # here with Carnot's handlers, and the gateway runs it.
    from OpenPinch.domain import configuration_fields

    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))
    monkeypatch.setattr(
        configuration_fields,
        "_REGISTERED_HPR_TYPES",
        set(configuration_fields._REGISTERED_HPR_TYPES),
    )
    carnot = registry.cycle_spec("carnot")
    registry.register_cycle(
        replace(
            carnot,
            id="stirling",
            label="Stirling cycle",
            topologies={"single": "Stirling cycle"},
            default_topology="single",
            handlers={"single": carnot.handlers["cascade"]},
            options_model=StirlingOptions,
            configuration_keys={},
        )
    )

    target = PinchProblem("basic_pinch.json").target.heat_pump_and_refrigeration(
        "stirling", **CHEAP
    )

    assert target.hpr_success
