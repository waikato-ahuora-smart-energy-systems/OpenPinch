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
    map_solution_to_network,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.fake_executor import (
    FakeSynthesisExecutor,
)
from OpenPinch.analysis.heat_exchanger_networks.models import (
    fixed_structure as fixed_model,
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

    assert structure.stage_count == 2
    assert structure.recovery == ((0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 1))
    assert structure.heaters == (0,)
    assert structure.coolers == (0, 1)
    assert structure.recovery_approach == {(1, 1, 1): 15.0}
    assert structure.hot_utility_approach == {0: 25.0}
    assert structure.cold_utility_approach == {1: 30.0}
    recovery, heaters, coolers = structure.z_restriction()
    assert recovery == [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
    assert heaters == [1, 0]
    assert coolers == [1, 1]


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

    assert structure.initial_recovery_duties == {(0, 0, 0): 1200.0}


def test_structure_accepts_qualified_stream_names() -> None:
    network = _four_stream_structure(
        recovery=(
            ("Process A.Raw Milk", "Process A.Milk Concentrate", 1),
            ("HT Flash", "CIP Water", 1),
        ),
        hot_utility="Hot Utility.HPS",
    )

    structure = fixed_network_structure(_request(network), AXIS_MAPS)

    assert structure.recovery == ((0, 0, 0), (1, 1, 0))


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
            "second heater",
        ),
        ({"coolers": ("Raw Milk", "Raw Milk", "HT Flash")}, "second cooler"),
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


# --- solution mapping --------------------------------------------------------


def _state(duty: float, *, active: bool | None = None) -> HeatExchangerPeriodState:
    return HeatExchangerPeriodState(
        period_id="0",
        period_idx=0,
        duty=duty,
        active=duty > 0.0 if active is None else active,
        approach_temperatures=(12.0, 11.0),
    )


def _solved(kind, source, sink, stage, duty, area) -> HeatExchanger:
    roles = {
        HeatExchangerKind.RECOVERY: (StreamID.Process, StreamID.Process),
        HeatExchangerKind.HOT_UTILITY: (StreamID.Utility, StreamID.Process),
        HeatExchangerKind.COLD_UTILITY: (StreamID.Process, StreamID.Utility),
    }[kind]
    return HeatExchanger(
        exchanger_id=f"solver-{source}-{sink}-{stage}",
        kind=kind,
        source_stream=source,
        sink_stream=sink,
        source_stream_role=roles[0],
        sink_stream_role=roles[1],
        stage=stage,
        period_states=(_state(duty),),
        area=area,
    )


def test_solution_mapping_keeps_listed_exchangers_and_user_ids() -> None:
    request = _request(
        network=_four_stream_structure(
            recovery=(
                ("Raw Milk", "Milk Concentrate", 1),
                ("HT Flash", "CIP Water", 1),
            ),
        ),
        min_approach_temperature=10.0,
    )
    structure = fixed_network_structure(request, AXIS_MAPS)
    recovery = HeatExchangerKind.RECOVERY
    extracted = HeatExchangerNetwork(
        exchangers=(
            _solved(
                recovery,
                "Process A.Raw Milk",
                "Process A.Milk Concentrate",
                1,
                2000.0,
                40.0,
            ),
            _solved(
                recovery, "Process A.Raw Milk", "Process A.CIP Water", 1, 0.0, None
            ),
            _solved(
                recovery, "Process A.HT Flash", "Process A.CIP Water", 1, 0.0, None
            ),
            _solved(
                recovery,
                "Process A.HT Flash",
                "Process A.Milk Concentrate",
                1,
                0.0,
                None,
            ),
            _solved(
                HeatExchangerKind.HOT_UTILITY,
                "Hot Utility.HPS",
                "Process A.Milk Concentrate",
                None,
                1600.0,
                10.0,
            ),
            _solved(
                HeatExchangerKind.COLD_UTILITY,
                "Process A.Raw Milk",
                "Cold Utility.CW",
                None,
                800.0,
                20.0,
            ),
            _solved(
                HeatExchangerKind.COLD_UTILITY,
                "Process A.HT Flash",
                "Cold Utility.CW",
                None,
                4400.0,
                60.0,
            ),
        ),
        total_annual_cost=1234.0,
        summary_metrics={"recovery_units": 4, "hot_utility_load": 1600.0},
        solver_axis_metadata={"axis_maps": AXIS_MAPS},
    )

    network = map_solution_to_network(request, structure, extracted)

    assert [exchanger.exchanger_id for exchanger in network.exchangers] == [
        exchanger.exchanger_id for exchanger in request.network.exchangers
    ]
    zero_duty = network.exchangers[1]
    assert zero_duty.source_stream == "Process A.HT Flash"
    assert zero_duty.period_states[0].duty == 0.0
    assert network.summary_metrics["recovery_units"] == 1
    assert network.summary_metrics["total_units"] == 4
    assert network.summary_metrics["total_area"] == pytest.approx(130.0)
    assert network.summary_metrics["duty_objective"] == "utility"
    assert network.summary_metrics["approach_temperature"] == 10.0
    assert network.objective_value == pytest.approx(1600.0 + 800.0 + 4400.0)

    area_request = _request(
        network=request.network, objective="area", max_hot_utility=2000.0
    )
    assert map_solution_to_network(
        area_request, structure, extracted
    ).objective_value == pytest.approx(130.0)
    cost_request = _request(network=request.network, objective="cost")
    assert map_solution_to_network(
        cost_request, structure, extracted
    ).objective_value == pytest.approx(1234.0)


def test_solution_mapping_fails_when_a_listed_exchanger_is_missing() -> None:
    request = _request(
        network=_four_stream_structure(
            recovery=(("Raw Milk", "Milk Concentrate", 1), ("HT Flash", "CIP Water", 1))
        )
    )
    structure = fixed_network_structure(request, AXIS_MAPS)
    extracted = HeatExchangerNetwork(
        exchangers=(),
        solver_axis_metadata={"axis_maps": AXIS_MAPS},
    )

    with pytest.raises(ValueError, match="solver result is missing"):
        map_solution_to_network(request, structure, extracted)


# --- model helpers (no GEKKO) ------------------------------------------------


class _RecordingModel:
    def __init__(self) -> None:
        self.equations: list = []
        self.minimised: list = []
        self.intermediates: list = []

    def Equation(self, expression):
        self.equations.append(expression)
        return expression

    def Minimize(self, expression):
        self.minimised.append(expression)

    def Intermediate(self, expression, *, name=None):
        self.intermediates.append((name, expression))
        return expression

    def sum(self, values):
        return sum(values)


def _owner(**overrides) -> SimpleNamespace:
    owner = SimpleNamespace(
        m=_RecordingModel(),
        tol=1e-3,
        N_periods=1,
        I=1,
        J=1,
        S=1,
        z_allowed=[[[1]]],
        z_hu_allowed=[1],
        z_cu_allowed=[1],
        dT_r_period=np.full((1, 1, 1), 5.0),
        dT_hu_period=np.full((1, 1), 5.0),
        dT_cu_period=np.full((1, 1), 5.0),
        default_approach=None,
        recovery_approach={},
        hot_utility_approach={},
        cold_utility_approach={},
        max_hot_utility=None,
        max_cold_utility=None,
        initial_recovery_duties={},
        hot_names=np.array(["H1"]),
        cold_names=np.array(["C1"]),
    )
    for key, value in overrides.items():
        setattr(owner, key, value)
    return owner


def test_approach_overrides_replace_contributions() -> None:
    owner = _owner(
        I=2,
        J=1,
        S=2,
        dT_r_period=np.full((1, 2, 1), 5.0),
        dT_hu_period=np.full((1, 1), 5.0),
        dT_cu_period=np.full((1, 2), 5.0),
        default_approach=10.0,
        recovery_approach={(0, 0, 0): 12.0, (0, 0, 1): 15.0},
        cold_utility_approach={1: 20.0},
    )

    fixed_model.apply_approach_overrides(owner)

    assert owner.dT_r_period[0].tolist() == [[12.0], [10.0]]
    assert owner.dT_hu_period[0].tolist() == [10.0]
    assert owner.dT_cu_period[0].tolist() == [10.0, 20.0]
    assert owner.dT_r.tolist() == [[12.0], [10.0]]


def test_stricter_stage_approach_adds_explicit_theta_bounds() -> None:
    owner = _owner(
        S=2,
        z_allowed=[[[1, 1]]],
        dT_r_period=np.full((1, 1, 1), 12.0),
        recovery_approach={(0, 0, 0): 12.0, (0, 0, 1): 15.0},
        theta_1_by_period=[[[[13.0, 16.0]]]],
        theta_2_by_period=[[[[13.0, 14.0]]]],
    )

    fixed_model.set_exchanger_approach_constraints(owner)

    assert owner.m.equations == [True, False]


def test_recovery_approach_definitions_tie_theta_to_end_differences() -> None:
    owner = _owner(
        non_isothermal_model=True,
        T_h_by_period=[[[400.0, 350.0]]],
        T_c_by_period=[[[380.0, 300.0]]],
        T_c_out_y_by_period=[[[[385.0]]]],
        T_h_out_x_by_period=[[[[340.0]]]],
        theta_1_by_period=[[[[15.0]]]],
        theta_2_by_period=[[[[40.0]]]],
    )

    fixed_model.set_recovery_approach_definitions(owner)

    assert owner.m.equations == [True, True]


def test_utility_approach_constraints_and_inlet_checks() -> None:
    owner = _owner(
        T_hu_in_period=np.array([[500.0]]),
        T_hu_out_period=np.array([[480.0]]),
        T_c_out_period=np.array([[490.0]]),
        T_c_by_period=[[[470.0, 300.0]]],
        T_cu_in_period=np.array([[290.0]]),
        T_cu_out_period=np.array([[300.0]]),
        T_h_out_period=np.array([[310.0]]),
        T_h_by_period=[[[400.0, 320.0]]],
    )

    fixed_model.set_utility_approach_constraints(owner)

    assert owner.m.equations == [True, True]

    owner.dT_hu_period = np.full((1, 1), 20.0)
    with pytest.raises(ValueError, match="Heater on cold stream C1"):
        fixed_model.set_utility_approach_constraints(owner)

    owner.dT_hu_period = np.full((1, 1), 5.0)
    owner.dT_cu_period = np.full((1, 1), 25.0)
    with pytest.raises(ValueError, match="Cooler on hot stream H1"):
        fixed_model.set_utility_approach_constraints(owner)


def test_utility_caps_skip_sides_without_utility_exchangers() -> None:
    owner = _owner(
        max_hot_utility=100.0,
        max_cold_utility=50.0,
        z_cu_allowed=[0],
        Q_h_by_period=[[80.0]],
        Q_c_by_period=[[0.0]],
    )

    fixed_model.set_utility_caps(owner)

    assert owner.m.equations == [True]


def test_total_area_objective_sums_listed_exchangers() -> None:
    owner = _owner(
        Q_r=[[[1000.0]]],
        U_r=np.array([[0.5]]),
        theta_1=[[[20.0]]],
        theta_2=[[[10.0]]],
        Q_h=[200.0],
        U_hu=np.array([0.8]),
        T_hu_in=np.array([500.0]),
        T_hu_out=np.array([500.0]),
        T_c_out=np.array([450.0]),
        T_c=[[440.0, 300.0]],
        Q_c=[300.0],
        U_cu=np.array([0.4]),
        T_h=[[400.0, 330.0]],
        T_h_out=np.array([320.0]),
        T_cu_in=np.array([290.0]),
        T_cu_out=np.array([300.0]),
    )

    fixed_model.set_total_area_objective(owner)

    def chen(a, b):
        return (a * b * (a + b) / 2 + 1e-3) ** (1 / 3)

    expected = (
        1000.0 / (0.5 * chen(20.0, 10.0))
        + 200.0 / (0.8 * chen(50.0, 60.0))
        + 300.0 / (0.4 * chen(30.0, 30.0))
    )
    assert owner.m.minimised == [pytest.approx(expected)]
    assert owner.total_area_expr == pytest.approx(expected)


def test_total_area_objective_is_single_period() -> None:
    with pytest.raises(ValueError, match="one operating period"):
        fixed_model.set_total_area_objective(_owner(N_periods=2))


def test_fixed_structure_model_rejects_other_goals_before_building() -> None:
    with pytest.raises(ValueError, match="supports minimisation goals"):
        fixed_model.FixedStructureStageWiseModel(minimisation_goal="hot utility")


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


# --- live solver --------------------------------------------------------------


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

    assert [exchanger.exchanger_id for exchanger in utility_network.exchangers] == [
        exchanger.exchanger_id for exchanger in structure.exchangers
    ]
    # Hot streams carry 7200 kW and cold streams 5550 kW.
    assert utility.total_cold_utility - utility.total_hot_utility == pytest.approx(
        1650.0, abs=1.0
    )
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
    structure = _four_stream_structure()
    strict_id = "recovery:HT Flash->CIP Water:S2"
    problem = _four_stream_problem()

    uniform = problem.design.optimise_duties(
        {
            "recovery": [
                ("Raw Milk", "Milk Concentrate", 1),
                ("HT Flash", "Milk Concentrate", 1),
                ("Raw Milk", "CIP Water", 2),
                ("HT Flash", "CIP Water", 2),
            ],
            "heaters": ["Milk Concentrate"],
            "coolers": ["Raw Milk", "HT Flash"],
        },
        objective="utility",
        min_approach_temperature=10.0,
    )
    strict = problem.design.optimise_duties(
        structure,
        objective="utility",
        min_approach_temperature=10.0,
        exchanger_approach_temperatures={strict_id: 40.0},
    )

    assert min(_recovery_approaches(strict.selected_network, strict_id)) >= 40.0 - 1e-2
    assert strict.total_hot_utility >= uniform.total_hot_utility - 1.0


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
