"""Fixed-structure HEN duty optimisation: request, mapping, model and solver tests."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.analysis.heat_exchanger_networks.duty_optimisation import (
    DEFAULT_FEASIBILITY_APPROACH,
    FixedStructureDutyExecutor,
    build_duty_optimisation_request,
    duty_optimisation_task,
    fixed_network_structure,
    heat_exchanger_network_duty_optimisation_service,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.fake_executor import (
    FakeSynthesisExecutor,
)
from OpenPinch.analysis.heat_exchanger_networks.models import (
    fixed_structure as fixed_model,
)
from OpenPinch.analysis.heat_exchanger_networks.models.fixed_structure import (
    COOLER,
    HEATER,
    RecoveryMatch,
    UtilityMatch,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
    require_solver_binary,
    require_synthesis_dependency,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.domain.enums import HeatExchangerKind, StreamID
from OpenPinch.domain.heat_exchanger import HeatExchanger
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from tests.support.paths import FIXTURES_ROOT
from tests.support.scipy_nlp import scipy_fixed_structure_model

FOUR_STREAM_FIXTURE = (
    FIXTURES_ROOT / "openhens" / "Four-stream-Yee-and-Grossmann-1990-1.json"
)

AXIS_MAPS = {
    "hot_process_streams": {"Process A.Raw Milk": 0, "Process A.HT Flash": 1},
    "cold_process_streams": {
        "Process A.Milk Concentrate": 0,
        "Process A.CIP Water": 1,
    },
    "hot_utilities": {"Hot Utility.HPS": 0},
    "cold_utilities": {"Cold Utility.CW": 0},
}


def _four_stream_structure(**overrides) -> HeatExchangerNetwork:
    arguments = {
        "recovery": (
            ("Raw Milk", "Milk Concentrate", 1),
            ("HT Flash", "Milk Concentrate", 1),
            ("Raw Milk", "CIP Water", 2),
            ("HT Flash", "CIP Water", 2),
        ),
        "heaters": ("Milk Concentrate",),
        "coolers": ("Raw Milk", "HT Flash"),
        "hot_utility": "HPS",
        "cold_utility": "CW",
    }
    arguments.update(overrides)
    return HeatExchangerNetwork.from_structure(**arguments)


def _request(network=None, **kwargs):
    kwargs.setdefault("objective", "utility")
    return build_duty_optimisation_request(
        network or _four_stream_structure(), **kwargs
    )


# --- structure builder -------------------------------------------------------


def test_from_structure_builds_zero_duty_exchangers_with_result_ids() -> None:
    network = _four_stream_structure(stage_count=3)

    assert network.stage_count == 3
    assert [exchanger.exchanger_id for exchanger in network.exchangers] == [
        "recovery:Raw Milk->Milk Concentrate:S1",
        "recovery:HT Flash->Milk Concentrate:S1",
        "recovery:Raw Milk->CIP Water:S2",
        "recovery:HT Flash->CIP Water:S2",
        "hot-utility:HPS->Milk Concentrate",
        "cold-utility:Raw Milk->CW",
        "cold-utility:HT Flash->CW",
    ]
    heater = network.exchangers[4]
    assert heater.kind is HeatExchangerKind.HOT_UTILITY
    assert heater.source_stream_role is StreamID.Utility
    assert all(
        state.duty == 0.0 and not state.active
        for exchanger in network.exchangers
        for state in exchanger.period_states
    )


def test_from_structure_requires_utility_names_for_utility_exchangers() -> None:
    with pytest.raises(ValueError, match="hot_utility is required"):
        HeatExchangerNetwork.from_structure(recovery=(("H", "C", 1),), heaters=("C",))
    with pytest.raises(ValueError, match="cold_utility is required"):
        HeatExchangerNetwork.from_structure(recovery=(("H", "C", 1),), coolers=("H",))


# --- request validation ------------------------------------------------------


def test_request_rejects_unknown_objective_and_non_network() -> None:
    with pytest.raises(ValueError, match="objective must be one of"):
        _request(objective="tac")
    with pytest.raises(TypeError, match="HeatExchangerNetwork"):
        build_duty_optimisation_request(object(), objective="utility")  # type: ignore[arg-type]


def test_area_objective_requires_a_utility_cap() -> None:
    with pytest.raises(ValueError, match="needs max_hot_utility"):
        _request(objective="area")

    request = _request(objective="area", max_cold_utility=2000)

    assert request.minimisation_goal == "total area"
    assert request.max_cold_utility == 2000.0
    assert request.max_hot_utility is None


@pytest.mark.parametrize("objective", ["utility", "cost"])
def test_utility_caps_only_apply_to_area_objective(objective: str) -> None:
    with pytest.raises(ValueError, match="'area' objective only"):
        _request(objective=objective, max_hot_utility=100.0)


def test_cost_objective_defaults_to_feasibility_approach() -> None:
    request = _request(objective="cost")

    assert request.minimisation_goal == "total cost"
    assert request.min_approach_temperature == DEFAULT_FEASIBILITY_APPROACH == 1.0
    assert (
        _request(objective="cost", min_approach_temperature=3).min_approach_temperature
        == 3.0
    )


def test_utility_objective_keeps_contributions_when_no_approach_given() -> None:
    request = _request()

    assert request.minimisation_goal == "total utility"
    assert request.min_approach_temperature is None


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"min_approach_temperature": 0.0}, ValueError, "must be positive"),
        ({"min_approach_temperature": float("nan")}, ValueError, "finite"),
        ({"min_approach_temperature": True}, TypeError, "must be a number"),
        (
            {"objective": "area", "max_hot_utility": -1.0},
            ValueError,
            "non-negative",
        ),
        (
            {"exchanger_approach_temperatures": {"missing": 5.0}},
            ValueError,
            "unknown exchangers: missing",
        ),
        (
            {"exchanger_approach_temperatures": [("x", 1.0)]},
            TypeError,
            "must map exchanger_id",
        ),
    ],
)
def test_request_rejects_invalid_numbers(kwargs, error, match) -> None:
    with pytest.raises(error, match=match):
        _request(**kwargs)


def test_request_records_json_safe_task_settings() -> None:
    request = _request(
        min_approach_temperature=10,
        exchanger_approach_temperatures={"recovery:Raw Milk->CIP Water:S2": 20},
    )

    settings = request.task_settings()

    assert json.loads(json.dumps(settings)) == settings
    assert settings["duty_objective"] == "utility"
    assert settings["exchanger_approach_temperatures"] == {
        "recovery:Raw Milk->CIP Water:S2": 20.0
    }


def test_request_needs_an_allowed_exchanger() -> None:
    network = _four_stream_structure()
    blocked = network.model_copy(
        update={
            "exchangers": tuple(
                exchanger.model_copy(update={"match_allowed": False})
                for exchanger in network.exchangers
            )
        }
    )

    with pytest.raises(ValueError, match="at least one allowed exchanger"):
        _request(blocked)


# --- structure mapping -------------------------------------------------------


def test_structure_maps_bare_stream_names_to_solver_indices() -> None:
    request = _request(
        exchanger_approach_temperatures={
            "recovery:HT Flash->CIP Water:S2": 15.0,
            "hot-utility:HPS->Milk Concentrate": 25.0,
            "cold-utility:HT Flash->CW": 30.0,
        }
    )

    structure = fixed_network_structure(request, AXIS_MAPS)
    spec = structure.spec

    assert spec.stage_count == 2
    assert spec.recovery == (
        RecoveryMatch(0, 0, 0),
        RecoveryMatch(1, 0, 0),
        RecoveryMatch(0, 1, 1),
        RecoveryMatch(1, 1, 1),
    )
    # Heaters default to the cold stream end (after stage 1), coolers to the
    # hot stream end (after the last stage).
    assert spec.utilities == (
        UtilityMatch(HEATER, 0, 0, 0),
        UtilityMatch(COOLER, 0, 0, 1),
        UtilityMatch(COOLER, 1, 0, 1),
    )
    assert spec.recovery_approach == {3: 15.0}
    assert spec.utility_approach == {0: 25.0, 2: 30.0}
    assert structure.positions == tuple(range(7))
    assert structure.slots == (
        ("recovery", 0),
        ("recovery", 1),
        ("recovery", 2),
        ("recovery", 3),
        ("utility", 0),
        ("utility", 1),
        ("utility", 2),
    )


def _utility_exchanger(
    kind: HeatExchangerKind,
    source: str,
    sink: str,
    stage: int | None = None,
    exchanger_id: str | None = None,
) -> HeatExchanger:
    heater = kind is HeatExchangerKind.HOT_UTILITY
    return HeatExchanger(
        exchanger_id=exchanger_id,
        kind=kind,
        source_stream=source,
        sink_stream=sink,
        source_stream_role=StreamID.Utility if heater else StreamID.Process,
        sink_stream_role=StreamID.Process if heater else StreamID.Utility,
        stage=stage,
        period_states=(
            HeatExchangerPeriodState(
                period_id="0", period_idx=0, duty=0.0, active=False
            ),
        ),
    )


def _with(network: HeatExchangerNetwork, *extra: HeatExchanger) -> HeatExchangerNetwork:
    return network.model_copy(update={"exchangers": network.exchangers + extra})


TWO_UTILITY_AXIS_MAPS = {
    **AXIS_MAPS,
    "hot_utilities": {"Hot Utility.LPS": 0, "Hot Utility.HPS": 1},
    "cold_utilities": {"Cold Utility.CW": 0, "Cold Utility.Refrigerant": 1},
}


def test_structure_places_several_utilities_and_mid_network_exchangers() -> None:
    network = _with(
        _four_stream_structure(),
        _utility_exchanger(
            HeatExchangerKind.HOT_UTILITY, "LPS", "Milk Concentrate", stage=2
        ),
        _utility_exchanger(HeatExchangerKind.HOT_UTILITY, "LPS", "Milk Concentrate"),
        _utility_exchanger(HeatExchangerKind.COLD_UTILITY, "Raw Milk", "CW", stage=1),
        _utility_exchanger(HeatExchangerKind.COLD_UTILITY, "Raw Milk", "Refrigerant"),
    )

    spec = fixed_network_structure(_request(network), TWO_UTILITY_AXIS_MAPS).spec

    assert spec.utilities == (
        UtilityMatch(HEATER, 0, 1, 0),  # HPS heater at the stream end
        UtilityMatch(COOLER, 0, 0, 1),
        UtilityMatch(COOLER, 1, 0, 1),
        UtilityMatch(HEATER, 0, 0, 1),  # LPS heater after stage 2
        UtilityMatch(HEATER, 0, 0, 0),  # LPS heater at the stream end
        UtilityMatch(COOLER, 0, 0, 0),  # CW cooler after stage 1
        UtilityMatch(COOLER, 0, 1, 1),  # refrigerant cooler at the stream end
    )


def test_structure_excludes_removed_positions_and_uses_warm_duties() -> None:
    structure = fixed_network_structure(
        _request(),
        AXIS_MAPS,
        excluded=frozenset({2, 5}),
        warm_duties={0: 1200.0, 4: 300.0},
    )

    assert structure.positions == (0, 1, 3, 4, 6)
    assert structure.spec.recovery == (
        RecoveryMatch(0, 0, 0),
        RecoveryMatch(1, 0, 0),
        RecoveryMatch(1, 1, 1),
    )
    assert structure.spec.utilities == (
        UtilityMatch(HEATER, 0, 0, 0),
        UtilityMatch(COOLER, 1, 0, 1),
    )
    assert structure.spec.initial_recovery_duties == {0: 1200.0}
    assert structure.spec.initial_utility_duties == {0: 300.0}


def test_structure_uses_existing_duties_as_warm_start() -> None:
    network = _four_stream_structure()
    first = network.exchangers[0]
    warm = first.model_copy(
        update={
            "period_states": (
                HeatExchangerPeriodState(period_id="0", period_idx=0, duty=1200.0),
            )
        }
    )
    network = network.model_copy(update={"exchangers": (warm, *network.exchangers[1:])})

    structure = fixed_network_structure(_request(network), AXIS_MAPS)

    assert structure.spec.initial_recovery_duties == {0: 1200.0}


def test_structure_accepts_qualified_stream_names() -> None:
    network = _four_stream_structure(
        recovery=(
            ("Process A.Raw Milk", "Process A.Milk Concentrate", 1),
            ("HT Flash", "CIP Water", 1),
        ),
        hot_utility="Hot Utility.HPS",
    )

    structure = fixed_network_structure(_request(network), AXIS_MAPS)

    assert structure.spec.recovery == (RecoveryMatch(0, 0, 0), RecoveryMatch(1, 1, 0))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {"recovery": (("Raw Milk", "Steam", 1),)},
            "'Steam' is not a cold process stream",
        ),
        ({"hot_utility": "LPS"}, "'LPS' is not a hot utility stream"),
        (
            {"recovery": (("CIP Water", "Raw Milk", 1),)},
            "'CIP Water' is not a hot process stream",
        ),
        (
            {
                "recovery": (
                    ("Raw Milk", "Milk Concentrate", 1),
                    ("Raw Milk", "Milk Concentrate", 1),
                    ("HT Flash", "CIP Water", 1),
                )
            },
            "duplicates another recovery exchanger",
        ),
        (
            {"heaters": ("Milk Concentrate", "Milk Concentrate")},
            "duplicates another exchanger with the same utility",
        ),
        (
            {"coolers": ("Raw Milk", "Raw Milk", "HT Flash")},
            "duplicates another exchanger with the same utility",
        ),
        (
            {"recovery": (("Raw Milk", "Milk Concentrate", 1),), "coolers": ()},
            "no exchanger serves: Process A.HT Flash, Process A.CIP Water",
        ),
        ({"stage_count": 1}, "stage_count is 1 but an exchanger is placed in stage 2"),
    ],
)
def test_structure_rejects_invalid_layouts(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        fixed_network_structure(
            _request(_four_stream_structure(**overrides)), AXIS_MAPS
        )


def test_utility_stage_counts_towards_the_stage_count() -> None:
    network = _with(
        _four_stream_structure(stage_count=2),
        _utility_exchanger(HeatExchangerKind.COLD_UTILITY, "Raw Milk", "CW", stage=3),
    )

    with pytest.raises(ValueError, match="placed in stage 3"):
        fixed_network_structure(_request(network), AXIS_MAPS)


def test_structure_rejects_ambiguous_bare_names() -> None:
    axis_maps = deepcopy(AXIS_MAPS)
    axis_maps["hot_process_streams"]["Process B.Raw Milk"] = 2

    with pytest.raises(ValueError, match="'Raw Milk' is ambiguous"):
        fixed_network_structure(_request(), axis_maps)


def test_task_records_structure_and_request() -> None:
    request = _request(min_approach_temperature=8.0)
    settings = SimpleNamespace(
        run_id="run",
        approach_temperatures=(10.0,),
        problem_id=None,
        workspace_variant=None,
        period_id=None,
    )

    task = duty_optimisation_task(request, settings)

    assert task.approach_temperature == 8.0
    assert task.stage_count == 2
    assert task.settings["fixed_structure"] is True
    assert task.seed_network == request.network
    assert len(task.topology_restrictions) == 4
    assert duty_optimisation_task(_request(), settings).approach_temperature == 10.0


# --- model helpers -----------------------------------------------------------


def test_utility_series_order() -> None:
    owner = SimpleNamespace(
        spec=fixed_model.FixedStructureSpec(
            stage_count=2,
            recovery=(),
            utilities=(
                UtilityMatch(HEATER, 0, 1, 0),  # HPS (600 K)
                UtilityMatch(HEATER, 0, 0, 0),  # LPS (450 K) runs first
                UtilityMatch(HEATER, 0, 0, 1),  # LPS between stages 2 and 1
                UtilityMatch(COOLER, 0, 0, 1),  # CW (300 K)
                UtilityMatch(COOLER, 0, 1, 1),  # refrigerant (250 K) runs last
            ),
        ),
        T_hu_in_period=np.array([[450.0, 600.0]]),
        T_cu_in_period=np.array([[300.0, 250.0]]),
    )

    assert fixed_model.utility_chains(owner) == {
        (HEATER, 0, 0): [1, 0],
        (HEATER, 0, 1): [2],
        (COOLER, 0, 1): [3, 4],
    }


def test_exact_lmtd_and_chen_agree_closely() -> None:
    assert fixed_model.exact_lmtd(20.0, 20.0) == 20.0
    assert fixed_model.exact_lmtd(30.0, 10.0) == pytest.approx(20.0 / np.log(3.0))
    assert fixed_model.exact_lmtd(0.0, 10.0) == 0.0
    assert fixed_model.chen_lmtd(30.0, 10.0) == pytest.approx(
        fixed_model.exact_lmtd(30.0, 10.0), rel=0.02
    )


def test_fixed_structure_model_rejects_other_goals_before_building() -> None:
    with pytest.raises(ValueError, match="supports minimisation goals"):
        fixed_model.FixedStructureModel(
            name="x",
            solver="apopt",
            solver_arrays=None,
            spec=fixed_model.FixedStructureSpec(1, (), ()),
            minimisation_goal="hot utility",
        )


# --- solved with the SciPy test backend (no GEKKO) ---------------------------


def _small_payload(*, hot_cp=10.0, extra_hot_utility=None) -> dict:
    payload = {
        "streams": [
            _stream("H1", hot_cp, 500.0, 300.0),
            _stream("C1", 5.0, 450.0, 650.0),
        ],
        "utilities": [
            _utility("HPS", "Hot", 700.0, 700.0, 80.0, 5.0),
            _utility("CW", "Cold", 290.0, 310.0, 15.0, 1.0),
        ],
        "options": {
            "COSTING_HX_AREA_COEFF": 150.0,
            "COSTING_HX_AREA_EXP": 1.0,
            "COSTING_HX_UNIT_COST": 5500.0,
            "HENS_APPROACH_TEMPERATURES": [10.0],
            "HENS_SOLVER_EVM": "ipopt-pyomo",
            "HENS_SOLVE_TOLERANCE": 0.001,
        },
    }
    if extra_hot_utility is not None:
        payload["utilities"].insert(1, extra_hot_utility)
    return payload


def _stream(name, cp, supply, target) -> dict:
    return {
        "name": name,
        "zone": "Site/Process A",
        "heat_capacity_flowrate": {"unit": "kW/delta_degC", "value": cp},
        "heat_flow": {"unit": "kW", "value": abs(cp * (target - supply))},
        "htc": {"unit": "kW/m^2/K", "value": 1.0},
        "t_supply": {"unit": "K", "value": supply},
        "t_target": {"unit": "K", "value": target},
    }


def _utility(name, kind, supply, target, price, htc) -> dict:
    return {
        "name": name,
        "type": kind,
        "heat_flow": None,
        "htc": {"unit": "kW/m^2/K", "value": htc},
        "price": {"unit": "$/MWh", "value": price},
        "t_supply": {"unit": "K", "value": supply},
        "t_target": {"unit": "K", "value": target},
    }


SMALL_STRUCTURE = {
    "recovery": [("H1", "C1", 1)],
    "heaters": ["C1"],
    "coolers": ["H1"],
    "hot_utility": "HPS",
}


@pytest.fixture
def scipy_backend(monkeypatch):
    import OpenPinch.analysis.heat_exchanger_networks.duty_optimisation as module

    factory = scipy_fixed_structure_model()
    original = module.FixedStructureDutyExecutor
    monkeypatch.setattr(
        module,
        "FixedStructureDutyExecutor",
        lambda request: original(request, model_factory=factory),
    )
    return factory


def _min_approach(network: HeatExchangerNetwork, kind=None) -> float:
    return min(
        approach
        for exchanger in network.exchangers
        if kind is None or exchanger.kind is kind
        for state in exchanger.period_states
        if state.active
        for approach in state.approach_temperatures
    )


def test_small_case_objectives_are_feasible_and_consistent(scipy_backend) -> None:
    problem = PinchProblem(_small_payload())

    utility = problem.design.optimise_duties(
        SMALL_STRUCTURE, objective="utility", min_approach_temperature=10.0
    )
    cost = problem.design.optimise_duties(SMALL_STRUCTURE, objective="cost")
    area = problem.design.optimise_duties(
        SMALL_STRUCTURE,
        objective="area",
        min_approach_temperature=10.0,
        max_hot_utility=900.0,
    )

    # H1 at 500 K can heat C1 (450 K, CP 5) to 490 K with 10 K approach.
    assert utility.total_heat_recovery == pytest.approx(200.0, abs=0.5)
    assert utility.total_hot_utility == pytest.approx(800.0, abs=0.5)
    assert _min_approach(utility.selected_network) >= 10.0 - 1e-3
    # A 1 K approach lets recovery rise and must not cost more.
    assert _min_approach(cost.selected_network) >= 1.0 - 1e-3
    assert cost.selected_network.total_annual_cost <= (
        utility.selected_network.total_annual_cost + 1.0
    )
    assert area.total_hot_utility <= 900.0 + 1e-3
    assert area.selected_network.summary_metrics["total_area"] <= (
        utility.selected_network.summary_metrics["total_area"] + 1e-3
    )
    for view in (utility, cost, area):
        network = view.selected_network
        assert network.total_annual_cost == pytest.approx(
            network.utility_cost + network.capital_cost
        )
        assert network.summary_metrics["removed_exchangers"] == ""


def test_two_hot_utilities_run_in_series_on_one_stream(scipy_backend) -> None:
    lps = _utility("LPS", "Hot", 600.0, 600.0, 40.0, 5.0)
    problem = PinchProblem(_small_payload(extra_hot_utility=lps))
    network = _with(
        HeatExchangerNetwork.from_structure(
            recovery=[("H1", "C1", 1)],
            heaters=["C1"],
            coolers=["H1"],
            hot_utility="HPS",
            cold_utility="CW",
        ),
        _utility_exchanger(
            HeatExchangerKind.HOT_UTILITY, "LPS", "C1", exchanger_id="lps"
        ),
    )

    view = problem.design.optimise_duties(
        network, objective="utility", min_approach_temperature=10.0
    )

    exchangers = {e.exchanger_id: e for e in view.selected_network.exchangers}
    lps_state = exchangers["lps"].period_states[0]
    hps_state = exchangers["hot-utility:HPS->C1"].period_states[0]
    assert lps_state.active and hps_state.active
    # LPS (colder) heats first; HPS finishes from where LPS stops.
    assert lps_state.sink_outlet_temperature == pytest.approx(
        hps_state.sink_inlet_temperature
    )
    assert lps_state.sink_outlet_temperature <= 600.0 - 10.0 + 1e-3
    assert hps_state.sink_outlet_temperature == pytest.approx(650.0)
    assert view.total_hot_utility == pytest.approx(800.0, abs=0.5)


def test_multi_period_area_uses_one_common_area(scipy_backend) -> None:
    payload = _small_payload()
    hot = payload["streams"][0]
    hot["heat_capacity_flowrate"] = {"unit": "kW/delta_degC", "values": [10.0, 8.0]}
    hot["heat_flow"] = {"unit": "kW", "values": [2000.0, 1600.0]}
    for record in payload["streams"][1:]:
        for key in ("heat_capacity_flowrate", "heat_flow"):
            record[key] = {
                "unit": record[key]["unit"],
                "values": [record[key]["value"]] * 2,
            }
    payload["options"].update(
        {"PROBLEM_PERIOD_IDS": ["high", "low"], "PROBLEM_PERIOD_WEIGHTS": [0.5, 0.5]}
    )
    problem = PinchProblem(payload)
    problem.target.all_periods.direct_heat_integration()

    view = problem.design.optimise_duties(
        {**SMALL_STRUCTURE, "hot_utility": "HPS"},
        objective="area",
        min_approach_temperature=10.0,
        max_hot_utility=900.0,
    )

    network = view.selected_network
    assert network.period_ids == ("high", "low")
    cooler = next(
        e for e in network.exchangers if e.kind is HeatExchangerKind.COLD_UTILITY
    )
    required = []
    for state in cooler.period_states:
        theta_1, theta_2 = state.approach_temperatures
        lmtd = fixed_model.exact_lmtd(theta_1, theta_2)
        required.append(state.duty / (0.5 * lmtd))
    # One area serves both periods: the larger requirement; the other period
    # runs with a bypass.
    assert cooler.area == pytest.approx(max(required), rel=1e-6)
    assert min(required) < cooler.area
    for state in network.exchangers[1].period_states:
        assert state.duty <= 900.0 + 1e-3


class _ScriptedModel:
    """Fake solved model: zero duty for the listed positions on each solve."""

    calls: list = []
    zero_by_call: list[set[int]] = []

    def __init__(self, *, spec, **_kwargs) -> None:
        self.spec = spec
        self.period_weights = [1.0]
        self.recovery_dt = [[1.0]]
        self.name = "scripted"
        self.solver_run = None
        type(self).calls.append(spec)

    def optimise(self, print_output=False) -> None:
        call = len(type(self).calls) - 1
        zero = type(self).zero_by_call[call]
        self.mSuccess = 1

        def result(index):
            duty = 0.0 if index in zero else 100.0
            period = fixed_model.ExchangerPeriodResult(
                duty=duty,
                active=duty > 0.0,
                approach=(10.0, 10.0),
                source_inlet=500.0,
                source_outlet=490.0,
                sink_inlet=400.0,
                sink_outlet=410.0,
                required_area=1.0,
            )
            return fixed_model.ExchangerResult([period], area=1.0, capital_cost=1.0)

        count = len(self.spec.recovery)
        self.recovery_results = [result(r) for r in range(count)]
        self.utility_results = [
            result(count + e) for e in range(len(self.spec.utilities))
        ]
        self.total_area = 1.0
        self.TAC = 2.0
        self.utility_cost_value = 1.0
        self.capital_cost_value = 1.0


def test_executor_removes_zero_duty_exchangers_and_resolves() -> None:
    problem = _four_stream_problem()
    problem.target.direct_heat_integration()
    request = _request(min_approach_temperature=10.0)
    task = duty_optimisation_task(
        request,
        SimpleNamespace(
            run_id="run",
            approach_temperatures=(10.0,),
            problem_id=None,
            workspace_variant=None,
            period_id=None,
        ),
    )
    _ScriptedModel.calls = []
    # Solve 1: slot 2 (recovery Raw Milk -> CIP Water) idle; solve 2: the
    # first cooler (slot 4 of the reduced spec) idle; solve 3: all carry duty.
    _ScriptedModel.zero_by_call = [{2}, {4}, set()]

    (outcome,) = FixedStructureDutyExecutor(
        request, model_factory=_ScriptedModel
    ).execute((task,), problem=problem, parent_outcomes={}, max_parallel=1)

    assert outcome.status == "success", outcome.error
    assert [
        len(spec.recovery) + len(spec.utilities) for spec in _ScriptedModel.calls
    ] == [7, 6, 5]
    network = outcome.network
    assert [e.exchanger_id for e in network.exchangers] == [
        "recovery:Raw Milk->Milk Concentrate:S1",
        "recovery:HT Flash->Milk Concentrate:S1",
        "recovery:HT Flash->CIP Water:S2",
        "hot-utility:HPS->Milk Concentrate",
        "cold-utility:HT Flash->CW",
    ]
    assert network.summary_metrics["removed_exchangers"] == (
        "recovery:Raw Milk->CIP Water:S2, cold-utility:Raw Milk->CW"
    )
    assert network.summary_metrics["removed_exchanger_count"] == 2
    # The re-solve starts from the previous duties.
    assert _ScriptedModel.calls[1].initial_recovery_duties == {
        0: 100.0,
        1: 100.0,
        2: 100.0,
    }


def test_executor_reports_solver_failure_after_removals() -> None:
    class Failing(_ScriptedModel):
        def optimise(self, print_output=False):
            if len(type(self).calls) == 2:
                self.mSuccess = 0
                self.solver_run = SimpleNamespace(
                    failure_reason="infeasible", status=None
                )
                return
            super().optimise(print_output)

    problem = _four_stream_problem()
    problem.target.direct_heat_integration()
    request = _request()
    task = duty_optimisation_task(
        request,
        SimpleNamespace(
            run_id="run",
            approach_temperatures=(10.0,),
            problem_id=None,
            workspace_variant=None,
            period_id=None,
        ),
    )
    Failing.calls = []
    Failing.zero_by_call = [{0}]

    (outcome,) = FixedStructureDutyExecutor(request, model_factory=Failing).execute(
        (task,), problem=problem, parent_outcomes={}, max_parallel=1
    )

    assert outcome.status == "failed"
    assert outcome.error == (
        "infeasible (after removing zero-duty exchangers: "
        "recovery:Raw Milk->Milk Concentrate:S1)"
    )


def test_four_stream_utility_objective_reaches_a_verified_network(
    scipy_backend,
) -> None:
    view = _four_stream_problem().design.optimise_duties(
        _four_stream_structure(), objective="utility", min_approach_temperature=10.0
    )

    network = view.selected_network
    # Pinch target at 10 K: 450 kW hot utility, 2100 kW cold utility.
    assert view.total_hot_utility >= 450.0 - 1.0
    assert view.total_cold_utility - view.total_hot_utility == pytest.approx(
        1650.0, abs=1.0
    )
    assert _min_approach(network) >= 10.0 - 1e-3
    assert network.summary_metrics["fixed_structure"] is True


# --- service wiring (fake executor) -----------------------------------------


def _four_stream_problem() -> PinchProblem:
    return PinchProblem(json.loads(FOUR_STREAM_FIXTURE.read_text(encoding="utf-8")))


def test_service_validates_before_targeting() -> None:
    problem = _four_stream_problem()

    with pytest.raises(ValueError, match="needs max_hot_utility"):
        heat_exchanger_network_duty_optimisation_service(
            problem, _four_stream_structure(), objective="area"
        )
    assert problem.results is None


def test_service_runs_one_recorded_task_through_executor() -> None:
    problem = _four_stream_problem()
    executor = FakeSynthesisExecutor()

    result = heat_exchanger_network_duty_optimisation_service(
        problem,
        _four_stream_structure(),
        objective="cost",
        executor=executor,
    )

    (task,) = executor.executed_tasks
    assert task.settings["duty_objective"] == "cost"
    assert task.settings["min_approach_temperature"] == 1.0
    assert task.stage_count == 2
    assert result.task_id == task.task_id
    assert problem.results.design.task_id == task.task_id


def test_public_accessor_returns_design_view_with_provenance(monkeypatch) -> None:
    import OpenPinch.analysis.heat_exchanger_networks.duty_optimisation as module

    executor = FakeSynthesisExecutor()
    monkeypatch.setattr(module, "FixedStructureDutyExecutor", lambda request: executor)
    problem = _four_stream_problem()

    view = problem.design.optimise_duties(
        _four_stream_structure(),
        objective="utility",
        min_approach_temperature=10.0,
    )

    assert view.result.provenance.method_id == "design.optimise_duties"
    assert executor.executed_tasks[0].settings["min_approach_temperature"] == 10.0
    assert problem.results.design.task_id == view.result.task_id


def test_executor_reports_structure_errors_as_failed_outcomes() -> None:
    problem = _four_stream_problem()
    problem.target.direct_heat_integration()
    request = _request(_four_stream_structure(hot_utility="LPS"))
    settings = SimpleNamespace(
        run_id="run",
        approach_temperatures=(10.0,),
        problem_id=None,
        workspace_variant=None,
        period_id=None,
    )
    task = duty_optimisation_task(request, settings)

    (outcome,) = FixedStructureDutyExecutor(request).execute(
        (task,), problem=problem, parent_outcomes={}, max_parallel=1
    )

    assert outcome.status == "failed"
    assert "'LPS' is not a hot utility stream" in outcome.error


# --- live solver (GEKKO) ------------------------------------------------------


def _skip_without_live_solver() -> None:
    try:
        require_synthesis_dependency("gekko", purpose="fixed-structure duty test")
        require_synthesis_dependency(
            "pyomo.environ", purpose="fixed-structure duty test"
        )
        require_solver_binary("ipopt", purpose="fixed-structure duty test")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError) as exc:
        pytest.skip(str(exc))


def _recovery_approaches(network: HeatExchangerNetwork, exchanger_id=None):
    return [
        approach
        for exchanger in network.exchangers
        if exchanger.kind is HeatExchangerKind.RECOVERY
        and (exchanger_id is None or exchanger.exchanger_id == exchanger_id)
        for state in exchanger.period_states
        if state.active
        for approach in state.approach_temperatures
    ]


@pytest.mark.synthesis
@pytest.mark.solver
def test_three_duty_objectives_on_four_stream_case() -> None:
    _skip_without_live_solver()
    structure = _four_stream_structure()
    problem = _four_stream_problem()

    utility = problem.design.optimise_duties(
        structure, objective="utility", min_approach_temperature=10.0
    )
    utility_network = utility.selected_network

    assert {e.exchanger_id for e in utility_network.exchangers} <= {
        e.exchanger_id for e in structure.exchangers
    }
    # Hot streams carry 7200 kW and cold streams 5550 kW; target 450 kW hot.
    assert utility.total_cold_utility - utility.total_hot_utility == pytest.approx(
        1650.0, abs=1.0
    )
    assert utility.total_hot_utility >= 450.0 - 1.0
    assert min(_recovery_approaches(utility_network)) >= 10.0 - 1e-2

    cap = 1.5 * utility.total_hot_utility + 100.0
    area = problem.design.optimise_duties(
        structure,
        objective="area",
        min_approach_temperature=10.0,
        max_hot_utility=cap,
    )
    assert area.total_hot_utility <= cap + 1.0
    assert area.selected_network.summary_metrics["total_area"] <= (
        utility_network.summary_metrics["total_area"] * 1.01
    )

    cost = problem.design.optimise_duties(structure, objective="cost")
    assert min(_recovery_approaches(cost.selected_network)) >= 1.0 - 1e-2
    assert cost.selected_network.total_annual_cost <= (
        utility_network.total_annual_cost * 1.01
    )


@pytest.mark.synthesis
@pytest.mark.solver
def test_per_exchanger_minimum_approach_is_enforced() -> None:
    _skip_without_live_solver()
    strict_id = "recovery:HT Flash->CIP Water:S2"
    problem = _four_stream_problem()
    structure = {
        "recovery": [
            ("Raw Milk", "Milk Concentrate", 1),
            ("HT Flash", "Milk Concentrate", 1),
            ("Raw Milk", "CIP Water", 2),
            ("HT Flash", "CIP Water", 2),
        ],
        "heaters": ["Milk Concentrate"],
        "coolers": ["Raw Milk", "HT Flash"],
    }

    uniform = problem.design.optimise_duties(
        structure, objective="utility", min_approach_temperature=10.0
    )
    strict = problem.design.optimise_duties(
        structure,
        objective="utility",
        min_approach_temperature=10.0,
        exchanger_approach_temperatures={strict_id: 40.0},
    )

    assert min(_recovery_approaches(strict.selected_network, strict_id)) >= 40.0 - 1e-2
    assert strict.total_hot_utility >= uniform.total_hot_utility - 1.0


@pytest.mark.synthesis
@pytest.mark.solver
def test_live_solver_runs_cheaper_utility_first() -> None:
    _skip_without_live_solver()
    lps = _utility("LPS", "Hot", 600.0, 600.0, 40.0, 5.0)
    problem = PinchProblem(_small_payload(extra_hot_utility=lps))
    network = _with(
        HeatExchangerNetwork.from_structure(
            recovery=[("H1", "C1", 1)],
            heaters=["C1"],
            coolers=["H1"],
            hot_utility="HPS",
            cold_utility="CW",
        ),
        _utility_exchanger(
            HeatExchangerKind.HOT_UTILITY, "LPS", "C1", exchanger_id="lps"
        ),
    )

    view = problem.design.optimise_duties(network, objective="cost")

    exchangers = {e.exchanger_id: e for e in view.selected_network.exchangers}
    lps_state = exchangers["lps"].period_states[0]
    assert lps_state.active
    assert lps_state.sink_outlet_temperature <= 600.0 - 1.0 + 1e-2
    assert view.total_hot_utility == pytest.approx(
        1000.0 - view.total_heat_recovery, abs=1.0
    )


@pytest.mark.synthesis
@pytest.mark.solver
def test_live_solver_enforces_approach_at_segment_boundaries() -> None:
    _skip_without_live_solver()

    view = _segmented().design.optimise_duties(
        SEGMENTED_STRUCTURE, objective="utility", min_approach_temperature=30.0
    )

    recovery = view.selected_network.exchangers[0]
    assert view.total_heat_recovery == pytest.approx(140.0, abs=0.5)
    assert min(_slice_approaches(recovery)) >= 30.0 - 0.1


def _capture_request(monkeypatch) -> list:
    import OpenPinch.analysis.heat_exchanger_networks.duty_optimisation as module

    requests: list = []

    def executor(request):
        requests.append(request)
        return FakeSynthesisExecutor()

    monkeypatch.setattr(module, "FixedStructureDutyExecutor", executor)
    return requests


def test_accessor_builds_structure_mapping_with_the_only_utilities(
    monkeypatch,
) -> None:
    requests = _capture_request(monkeypatch)
    problem = _four_stream_problem()

    problem.design.optimise_duties(
        {
            "recovery": [
                ("Raw Milk", "Milk Concentrate", 1),
                ("HT Flash", "CIP Water", 1),
            ],
            "heaters": ["CIP Water"],
            "coolers": ["HT Flash", "Raw Milk"],
        },
        objective="cost",
    )

    (request,) = requests
    assert [exchanger.exchanger_id for exchanger in request.network.exchangers] == [
        "recovery:Raw Milk->Milk Concentrate:S1",
        "recovery:HT Flash->CIP Water:S1",
        "hot-utility:HPS->CIP Water",
        "cold-utility:HT Flash->CW",
        "cold-utility:Raw Milk->CW",
    ]


@pytest.mark.parametrize(
    ("structure", "match"),
    [
        ({"recovery": [], "splits": []}, "got splits"),
        ({"heaters": ["CIP Water"]}, "needs a 'recovery' list"),
    ],
)
def test_accessor_rejects_malformed_structure_mappings(structure, match) -> None:
    with pytest.raises(ValueError, match=match):
        _four_stream_problem().design.optimise_duties(structure)


def test_accessor_needs_explicit_utility_when_ambiguous(monkeypatch) -> None:
    requests = _capture_request(monkeypatch)
    payload = json.loads(FOUR_STREAM_FIXTURE.read_text(encoding="utf-8"))
    second = deepcopy(payload["utilities"][0])
    second["name"] = "LPS"
    payload["utilities"].append(second)
    structure = {
        "recovery": [("Raw Milk", "Milk Concentrate", 1), ("HT Flash", "CIP Water", 1)],
        "heaters": ["CIP Water"],
        "coolers": ["HT Flash", "Raw Milk"],
    }

    with pytest.raises(ValueError, match="pass hot_utility explicitly"):
        PinchProblem(payload).design.optimise_duties(structure)

    PinchProblem(payload).design.optimise_duties({**structure, "hot_utility": "LPS"})
    assert requests[0].network.exchangers[2].source_stream == "LPS"


def test_utility_that_cannot_reach_its_stream_is_reported(scipy_backend) -> None:
    cold_steam = _utility("LPS", "Hot", 455.0, 455.0, 40.0, 5.0)
    problem = PinchProblem(_small_payload(extra_hot_utility=cold_steam))
    network = _with(
        HeatExchangerNetwork.from_structure(
            recovery=[("H1", "C1", 1)],
            heaters=["C1"],
            coolers=["H1"],
            hot_utility="HPS",
            cold_utility="CW",
        ),
        _utility_exchanger(HeatExchangerKind.HOT_UTILITY, "LPS", "C1"),
    )

    with pytest.raises(Exception, match="heater using .*LPS on cold stream C1"):
        problem.design.optimise_duties(
            network, objective="utility", min_approach_temperature=10.0
        )


def test_from_structure_accepts_named_and_staged_utility_entries() -> None:
    network = HeatExchangerNetwork.from_structure(
        recovery=[("H1", "C1", 1), ("H1", "C1", 2)],
        heaters=["C1", ("C1", "LPS", 2)],
        coolers=[("H1", "CW"), ("H1", "Chilled", 2)],
        hot_utility="HPS",
    )

    assert [(e.exchanger_id, e.stage) for e in network.exchangers[2:]] == [
        ("hot-utility:HPS->C1", None),
        ("hot-utility:LPS->C1:S2", 2),
        ("cold-utility:H1->CW", None),
        ("cold-utility:H1->Chilled:S2", 2),
    ]
    with pytest.raises(ValueError, match="cold_utility is required"):
        HeatExchangerNetwork.from_structure(recovery=[], coolers=["H1"])
    with pytest.raises(ValueError, match="entries must be a stream name"):
        HeatExchangerNetwork.from_structure(recovery=[], heaters=[("C1",)])


def test_accessor_structure_with_named_utilities_needs_no_default(
    monkeypatch,
) -> None:
    requests = _capture_request(monkeypatch)
    payload = json.loads(FOUR_STREAM_FIXTURE.read_text(encoding="utf-8"))
    second = deepcopy(payload["utilities"][0])
    second["name"] = "LPS"
    payload["utilities"].append(second)

    PinchProblem(payload).design.optimise_duties(
        {
            "recovery": [
                ("Raw Milk", "Milk Concentrate", 1),
                ("HT Flash", "CIP Water", 1),
            ],
            "heaters": [("CIP Water", "LPS"), ("CIP Water", "HPS")],
            "coolers": ["HT Flash", "Raw Milk"],
        }
    )

    heaters = [
        e.exchanger_id
        for e in requests[0].network.exchangers
        if e.kind is HeatExchangerKind.HOT_UTILITY
    ]
    assert heaters == ["hot-utility:LPS->CIP Water", "hot-utility:HPS->CIP Water"]


# --- segmented streams and utilities ------------------------------------------

SEGMENTED_STRUCTURE = {
    "recovery": [("Hot parent", "Cold parent", 1)],
    "heaters": ["Cold parent"],
    "coolers": ["Hot parent"],
}


def _segmented(**kwargs) -> PinchProblem:
    from tests.analysis.heat_exchanger_networks.test_segmented_streams import (
        _segmented_problem,
    )

    return _segmented_problem(**kwargs)


def _slice_approaches(exchanger: HeatExchanger) -> list[float]:
    return [
        delta
        for item in exchanger.segment_area_contributions
        for delta in (
            item.hot_inlet_temperature - item.cold_outlet_temperature,
            item.hot_outlet_temperature - item.cold_inlet_temperature,
        )
    ]


def test_segmented_streams_respect_the_approach_inside_the_exchanger(
    scipy_backend,
) -> None:
    # Hot 200 -> 150 (CP 1) -> 100 (CP 2); cold 50 -> 100 (CP 1) -> 150 (CP 2).
    # Both ends keep 50 K at full recovery (150 kW), but the segment kinks
    # pinch inside the exchanger: 30 K everywhere allows only 140 kW.
    view = _segmented().design.optimise_duties(
        SEGMENTED_STRUCTURE, objective="utility", min_approach_temperature=30.0
    )

    network = view.selected_network
    recovery = network.exchangers[0]
    assert view.total_heat_recovery == pytest.approx(140.0, abs=0.5)
    assert view.total_hot_utility == pytest.approx(10.0, abs=0.5)
    assert recovery.segment_area_contributions
    assert min(_slice_approaches(recovery)) >= 30.0 - 0.1
    assert recovery.area == pytest.approx(
        sum(s.area for s in recovery.segment_area_contributions)
    )


def test_segmented_streams_without_a_binding_kink_recover_fully(scipy_backend) -> None:
    view = _segmented().design.optimise_duties(
        SEGMENTED_STRUCTURE, objective="utility", min_approach_temperature=10.0
    )

    network = view.selected_network
    assert view.total_heat_recovery == pytest.approx(150.0, abs=0.5)
    # Heater and cooler are idle, so both are removed and the case re-solved.
    assert network.summary_metrics["removed_exchanger_count"] == 2
    assert [e.kind for e in network.exchangers] == [HeatExchangerKind.RECOVERY]


def test_segmented_utility_draws_from_its_profile(scipy_backend) -> None:
    # HU: 250 -> 225 C at 20 $/MWh, then 225 -> 200 C at 80; the larger cold
    # load (50 -> 100 C CP 1, 100 -> 150 C CP 3) needs hot utility.
    view = _segmented(
        segmented_utility=True, cold_second_duty=150.0
    ).design.optimise_duties(
        SEGMENTED_STRUCTURE, objective="utility", min_approach_temperature=30.0
    )

    network = view.selected_network
    heater = next(
        e for e in network.exchangers if e.kind is HeatExchangerKind.HOT_UTILITY
    )
    state = heater.period_states[0]
    assert state.active
    assert state.source_inlet_temperature == pytest.approx(523.15)
    assert 473.15 - 1e-6 <= state.source_outlet_temperature <= 523.15
    assert state.source_split_fraction is None
    assert view.total_heat_recovery == pytest.approx(140.0, abs=0.5)
    assert state.duty == pytest.approx(60.0, abs=0.5)
    assert network.utility_cost > 0.0
    assert heater.segment_area_contributions
    assert min(_slice_approaches(heater)) >= 30.0 - 0.1
    assert network.total_annual_cost == pytest.approx(
        network.utility_cost + network.capital_cost
    )
