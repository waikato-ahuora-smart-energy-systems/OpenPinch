"""Focused edge-path tests for HEN synthesis support helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import OpenPinch.analysis.heat_exchanger_networks.models.problem as unit_problem
import OpenPinch.analysis.heat_exchanger_networks.models.stagewise as stagewise
from OpenPinch.analysis.heat_exchanger_networks import (
    service as entry,
)
from OpenPinch.analysis.heat_exchanger_networks.context import (
    finalise_design_result,
    optional_text,
)
from OpenPinch.analysis.heat_exchanger_networks.errors import (
    WorkflowContractError,
)
from OpenPinch.analysis.heat_exchanger_networks.execution import (
    fallbacks,
    pathways,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.fake_executor import (
    FakeSynthesisExecutor,
    _fake_network,
    _first_stream,
    _temperature,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.pathways import (
    pathways_from_metadata,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.settings import (
    SynthesisWorkflowSettings,
)
from OpenPinch.analysis.heat_exchanger_networks.models._stagewise import (
    verification as stagewise_verification,
)
from OpenPinch.analysis.heat_exchanger_networks.models.problem import (
    InternalHeatExchangerNetworkProblem,
)
from OpenPinch.analysis.heat_exchanger_networks.reporting import (
    exports,
    ranking,
)
from OpenPinch.analysis.heat_exchanger_networks.results import (
    assembly,
    seeds,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import (
    dependencies,
)
from OpenPinch.analysis.heat_exchanger_networks.targeting import (
    network_evolution_method as evolution,
)
from OpenPinch.analysis.heat_exchanger_networks.targeting import (
    open_hens_method,
)
from OpenPinch.analysis.heat_exchanger_networks.targeting.topology import (
    canonical_stage_count,
    canonical_topology_restrictions,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.contracts.synthesis.common import (
    HeatExchangerNetworkSynthesisExportRecord,
    HeatExchangerNetworkSynthesisManifest,
)
from OpenPinch.contracts.synthesis.task import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from OpenPinch.domain._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.domain.enums import (
    HeatExchangerKind,
    StreamID,
)
from OpenPinch.domain.heat_exchanger import HeatExchanger
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork


def _task(
    *,
    method: str = "pinch_design_method",
    task_id: str = "task-1",
    parent_task_id: str | None = None,
    metadata: dict | None = None,
) -> HeatExchangerNetworkSynthesisTask:
    return HeatExchangerNetworkSynthesisTask(
        task_id=task_id,
        run_id="run-1",
        method=method,
        approach_temperature=10.0,
        derivative_threshold=0.5,
        stage_count=1,
        parent_task_id=parent_task_id,
        metadata=metadata or {},
    )


def _outcome(
    *,
    method: str = "pinch_design_method",
    status: str = "failed",
    task_id: str = "task-1",
    error: str | None = None,
    network: HeatExchangerNetwork | None = None,
) -> HeatExchangerNetworkSynthesisTaskOutcome:
    return HeatExchangerNetworkSynthesisTaskOutcome(
        task=_task(method=method, task_id=task_id),
        status=status,
        error=error,
        solver_status="solver-status",
        objective_value=1.0 if status == "success" else None,
        network=network,
    )


def _settings(**overrides):
    values = {
        "run_id": "run-1",
        "max_parallel": 1,
        "problem_id": "problem-1",
        "workspace_variant": "baseline",
        "period_id": None,
        "synthesis_quality_tier": 1,
        "approach_temperatures": (10.0,),
        "stage_selection": (1,),
        "evm_n_ad_branches": None,
        "evm_n_rm_branches": None,
        "tdm_solver": "couenne",
        "pdm_solver": "couenne",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _Variable:
    def __init__(self, lower: float = -1e9, upper: float = 1e9):
        self.lower = lower
        self.upper = upper
        self.VALUE = SimpleNamespace(value=None)


class _IndexRejectingElement:
    def __setitem__(self, _index, _value):
        raise IndexError("scalar")


def test_fake_executor_rejects_missing_parent_and_empty_problem_inputs():
    executor = FakeSynthesisExecutor()
    child_task = _task(
        method="network_evolution_method",
        task_id="child",
        parent_task_id="missing-parent",
    )

    with pytest.raises(WorkflowContractError, match="successful parent outcomes"):
        executor.execute(
            (child_task,),
            problem=SimpleNamespace(master_zone=object()),
            parent_outcomes={},
            max_parallel=1,
        )

    with pytest.raises(RuntimeError, match="prepared root Zone"):
        _fake_network(SimpleNamespace(master_zone=None), _task())

    with pytest.raises(ValueError, match="at least one hot process stream"):
        _first_stream([], "hot process")

    assert _temperature(None, "K") is None
    assert _temperature(350.0, "K") == pytest.approx(350.0)


def test_couenne_fallback_predicates_reject_non_matching_inputs():
    task = _task()
    successful = _outcome(status="success", network=HeatExchangerNetwork())
    successful_tdm = _outcome(
        method="thermal_derivative_method",
        status="success",
        network=HeatExchangerNetwork(),
    )
    non_couenne = _settings(tdm_solver="ipopt", pdm_solver="ipopt")

    assert (
        fallbacks._can_skip_derivative_stage_for_missing_couenne(
            non_couenne,
            (task,),
            (successful,),
        )
        is False
    )
    assert (
        fallbacks._can_skip_preliminary_stages_for_missing_couenne(
            non_couenne,
            (task,),
            (successful,),
        )
        is False
    )
    assert (
        fallbacks._can_skip_derivative_stage_for_missing_couenne(
            _settings(),
            (task,),
            (successful_tdm,),
        )
        is False
    )
    assert fallbacks._missing_couenne_failure(successful) is False
    assert (
        fallbacks._missing_couenne_failure(
            _outcome(
                method="network_evolution_method", error="couenne not found on PATH"
            )
        )
        is False
    )


def test_pathway_metadata_skips_invalid_items_and_uses_explicit_breadth():
    metadata = {
        "pathways": [
            "not-a-pathway",
            {
                "pathway_id": "p1",
                "tier_origin": 2,
                "pathway_kind": "quality",
                "pdm_mode": "compact",
                "pdm_multiplier": None,
                "uses_tdm": False,
                "evm_n_ad_branches": 2,
                "evm_n_rm_branches": 3,
                "evm_no_improvement_patience": None,
                "protected": False,
            },
        ]
    }

    rehydrated = pathways_from_metadata(metadata)

    assert len(rehydrated) == 1
    assert rehydrated[0].pathway_id == "p1"
    assert pathways._branch_breadth(0, tier=5) == 1


def test_export_ranking_and_result_summary_edge_paths(monkeypatch, tmp_path: Path):
    with pytest.raises(TypeError, match="live PinchProblem"):
        exports.export_heat_exchanger_network_synthesis_results(object(), tmp_path)

    problem = object.__new__(PinchProblem)
    problem._results = None
    with pytest.raises(RuntimeError, match="Run problem.design"):
        exports.export_heat_exchanger_network_synthesis_results(problem, tmp_path)

    multiperiod_network = HeatExchangerNetwork(
        exchangers=(
            HeatExchanger(
                exchanger_id="H1-C1-1",
                kind=HeatExchangerKind.RECOVERY,
                source_stream="H1",
                sink_stream="C1",
                source_stream_role=StreamID.Process,
                sink_stream_role=StreamID.Process,
                stage=1,
                period_states=(
                    HeatExchangerPeriodState(period_id="base", period_idx=0, duty=10.0),
                    HeatExchangerPeriodState(period_id="peak", period_idx=1, duty=20.0),
                ),
            ),
        )
    )
    problem._results = SimpleNamespace(
        design=SimpleNamespace(network=multiperiod_network)
    )
    with pytest.raises(ValueError, match="period_id is required"):
        exports.export_heat_exchanger_network_synthesis_results(problem, tmp_path)

    design = SimpleNamespace(
        manifest=None,
        run_id="run-1",
        ranked_networks=(_outcome(status="success", network=HeatExchangerNetwork()),),
        stage_count=None,
        period_id="annual",
    )
    manifest = exports._manifest_with_export_records(
        design,
        records=(
            HeatExchangerNetworkSynthesisExportRecord(
                run_id="run-1",
                format="json",
                path="manifest.json",
            ),
        ),
        period_id="annual",
        problem_id="problem-1",
        workspace_variant="baseline",
    )
    assert isinstance(manifest, HeatExchangerNetworkSynthesisManifest)

    with pytest.raises(ValueError, match="limit"):
        ranking.rank_unique_network_outcomes(
            SimpleNamespace(ranked_networks=()), limit=0
        )

    outcomes = (
        _outcome(status="success", task_id="a", network=HeatExchangerNetwork()),
        _outcome(status="success", task_id="b", network=HeatExchangerNetwork()),
    )
    monkeypatch.setattr(ranking, "_ranked_candidate_outcomes", lambda _result: outcomes)
    monkeypatch.setattr(
        ranking,
        "network_structure_signature",
        lambda network: ((id(network),),),
    )
    assert len(ranking.rank_unique_network_outcomes(object(), limit=1)) == 1

    assert assembly._failed_outcome_summary(()) == ""
    failed_summary = assembly._failed_outcome_summary(
        tuple(
            _outcome(task_id=f"failed-{idx}", error=f"error-{idx}") for idx in range(4)
        )
    )
    assert "1 more failed task" in failed_summary
    assert assembly._infeasible_outcome_summary(()) == ""
    infeasible_summary = assembly._infeasible_outcome_summary(
        tuple((_outcome(task_id=f"bad-{idx}"), (f"failure-{idx}",)) for idx in range(4))
    )
    assert "1 more infeasible task" in infeasible_summary


def test_result_building_and_seed_validation_edges(monkeypatch):
    monkeypatch.setattr(assembly, "verify_network_feasibility", lambda _network: ())
    with pytest.raises(WorkflowContractError, match="must include a network"):
        assembly.build_synthesis_result(
            _settings(
                derivative_thresholds=(0.5,),
                quality_derivative_thresholds=(0.5,),
                method_sequence=("pinch_design_method",),
                output_formats=("json",),
                solve_tolerance=1e-6,
                best_solutions_to_save=1,
                pdm_stage_pair_limit=None,
                tdm_parent_limit=None,
                stage_packing="auto",
                effective_evm_n_ad_branches=1,
                effective_evm_n_rm_branches=1,
                design_method="open_hens_method",
                solver_for=lambda _method: "fake",
            ),
            [_task()],
            [_outcome(status="success", network=None)],
        )

    seed_network = HeatExchangerNetwork()
    assert seeds._normalise_seed_networks(seed_network) == (seed_network,)
    with pytest.raises(ValueError, match="at least one network"):
        seeds._normalise_seed_networks(())
    with pytest.raises(TypeError, match="HeatExchangerNetwork"):
        seeds._normalise_seed_networks((object(),))

    problem = SimpleNamespace(
        results=SimpleNamespace(
            design=SimpleNamespace(method="pinch_design_method", network=seed_network)
        )
    )
    with pytest.raises(ValueError, match="cached thermal_derivative_method"):
        seeds.resolve_seed_networks(
            problem,
            None,
            method_name="network_evolution_method",
            cached_source_method="thermal_derivative_method",
        )


def test_service_context_solver_dependency_and_entry_guards(monkeypatch, tmp_path):
    class FakeDesign:
        ranked_networks = ()
        network = SimpleNamespace()

        def model_copy(self, update):
            self.update = update
            return self

    monkeypatch.setattr(
        "OpenPinch.analysis.heat_exchanger_networks.context.rank_unique_network_outcomes",
        lambda _result: (),
    )
    monkeypatch.setattr(
        "OpenPinch.analysis.heat_exchanger_networks.context.verify_synthesis_result",
        lambda _design: ("bad design",),
    )
    with pytest.raises(RuntimeError, match="verification failed"):
        finalise_design_result(
            SimpleNamespace(),
            SimpleNamespace(model_copy=lambda update: SimpleNamespace()),
            SimpleNamespace(accepted_result=FakeDesign()),
        )

    assert optional_text(12) == "12"

    fake_module = object()
    monkeypatch.setattr(dependencies, "SYNTHESIS_DEPENDENCIES", ())
    assert dependencies.require_declared_synthesis_dependencies() == {}
    monkeypatch.setattr(
        dependencies,
        "import_module",
        lambda name: (
            fake_module if name != "idaes" else (_ for _ in ()).throw(ImportError)
        ),
    )
    assert dependencies._idaes_bin_directory() is None
    monkeypatch.setattr(
        dependencies,
        "import_module",
        lambda name: SimpleNamespace(bin_directory=None),
    )
    assert dependencies._idaes_bin_directory() is None
    monkeypatch.setattr(
        dependencies,
        "import_module",
        lambda name: SimpleNamespace(bin_directory=str(tmp_path)),
    )
    assert dependencies._idaes_solver_binary("missing-solver") is None

    with pytest.raises(ValueError, match="open_hens_method does not accept"):
        entry.heat_exchanger_network_synthesis_service(object(), initial_networks=())

    monkeypatch.setattr(entry, "_coerce_design_method", lambda _method: object())
    with pytest.raises(AssertionError, match="Unhandled HEN design method"):
        entry.heat_exchanger_network_synthesis_service(object(), method="unknown")


def test_network_evolution_task_edges(monkeypatch):
    class FakeExecutor:
        def execute(self, tasks, *, problem, parent_outcomes, max_parallel):
            return tuple(
                HeatExchangerNetworkSynthesisTaskOutcome(
                    task=task,
                    status="success",
                    network=HeatExchangerNetwork(stage_count=task.stage_count),
                    objective_value=1.0,
                )
                for task in tasks
            )

    monkeypatch.setattr(evolution, "LocalSynthesisExecutor", FakeExecutor)
    settings = _settings()
    assert evolution.execute_network_evolution_method_stage(
        problem=object(),
        settings=settings,
        tdm_outcomes=(),
        parent_outcomes={},
    ) == ((), ())
    assert evolution.execute_network_evolution_method_from_pinch_design_stage(
        problem=object(),
        settings=settings,
        pdm_outcomes=(),
        parent_outcomes={},
    ) == ((), ())
    direct_tasks, direct_outcomes = (
        evolution.execute_direct_network_evolution_method_stage(
            problem=object(),
            settings=settings,
        )
    )
    assert len(direct_tasks) == len(direct_outcomes) == 1
    assert (
        evolution.build_network_evolution_method_tasks_from_pinch_design_method(
            (_outcome(status="failed"),),
            settings=settings,
        )
        == ()
    )

    monkeypatch.setattr(
        evolution, "required_topology_restrictions_from_outcome", lambda *_args: ()
    )
    monkeypatch.setattr(evolution, "_required_stage_count", lambda *_args: 1)
    standard_tasks = evolution.build_network_evolution_method_tasks(
        settings,
        (
            _outcome(
                method="thermal_derivative_method",
                status="success",
                network=HeatExchangerNetwork(),
            ),
        ),
    )
    assert len(standard_tasks) == 1

    assert evolution._evolution_task_settings(
        n_ad_branches=2,
        n_rm_branches=3,
        no_improvement_patience=4,
    ) == {
        "evolution_n_ad_branches": 2,
        "evolution_n_rm_branches": 3,
        "evolution_no_improvement_patience": 4,
    }

    monkeypatch.setattr(
        evolution, "topology_restrictions_from_network", lambda *_args, **_kwargs: ()
    )
    monkeypatch.setattr(
        evolution, "canonical_topology_restrictions", lambda restrictions: ()
    )
    monkeypatch.setattr(
        evolution, "approach_temperature_from_network", lambda *_args: 10.0
    )
    monkeypatch.setattr(
        evolution, "topology_restriction_signature", lambda _restrictions: ()
    )
    monkeypatch.setattr(evolution, "canonical_stage_count", lambda _restrictions: 1)
    monkeypatch.setattr(
        evolution, "derivative_threshold_from_network", lambda _network: None
    )

    seeded_tasks = evolution.build_seeded_network_evolution_method_tasks(
        _settings(synthesis_quality_tier=2),
        (HeatExchangerNetwork(), HeatExchangerNetwork()),
    )

    assert len(seeded_tasks) == 1


def test_topology_helpers_handle_empty_restrictions():
    assert canonical_topology_restrictions(()) == ()
    with pytest.raises(ValueError, match="at least one match"):
        canonical_stage_count(())


def test_open_hens_rejects_modified_method_sequence():
    settings = SynthesisWorkflowSettings(
        run_id="open-hens",
        approach_temperatures=(10.0,),
        derivative_thresholds=(0.5,),
        stage_selection=(1,),
        method_sequence=("pinch_design_method", "network_evolution_method"),
        output_formats=(),
        solve_tolerance=1e-3,
        best_solutions_to_save=1,
        max_parallel=1,
        pdm_solver="couenne",
        tdm_solver="couenne",
        pdm_solver_options={},
        tdm_solver_options={},
        evm_solver="ipopt",
        evm_solver_options={},
    )

    with pytest.raises(WorkflowContractError, match="must preserve"):
        open_hens_method.execute_open_hens_method(object(), settings)


def test_internal_problem_error_and_pdm_print_paths():
    pdm_problem = InternalHeatExchangerNetworkProblem(
        solver_arrays=object(),
        framework="PDM",
    )
    with pytest.raises(ValueError, match="above and below pinch"):
        pdm_problem.load_model(model_factories={})

    stage_problem = InternalHeatExchangerNetworkProblem(
        solver_arrays=object(),
        framework="TDM",
    )
    with pytest.raises(ValueError, match="explicit stage count"):
        stage_problem.load_model(model_factories={})
    with pytest.raises(
        ValueError, match="Unknown heat exchanger network model factory"
    ):
        stage_problem._model_factory(None, "unknown", default="UnknownModel")

    def raising_stagewise_factory(**_kwargs):
        raise ValueError("bad model")

    failing_problem = InternalHeatExchangerNetworkProblem(
        solver_arrays=object(),
        framework="TDM",
        stages=1,
    )
    assert (
        failing_problem.get_solution(
            print_output=False,
            model_factories={"stagewise": raising_stagewise_factory},
        )
        is None
    )
    assert failing_problem.solution_failure_reason == "bad model"

    class FakeSide:
        side_required = True

        def __init__(self):
            self.optimised = False

        def optimise(self, print_output=True):
            self.optimised = True

        def amalgamate_networks(self, *, below_case, above_case):
            return FakeAmalgamatedCase()

    class FakeAmalgamatedCase:
        S = 1

        def __init__(self):
            self.post_processed = False
            self.output_called = False

        def get_post_process(self):
            self.post_processed = True

        def output_to_cmd_line(self):
            self.output_called = True

    solve_problem = InternalHeatExchangerNetworkProblem(
        solver_arrays=object(),
        framework="PDM",
    )
    solve_problem.above = FakeSide()
    solve_problem.below = FakeSide()
    solve_problem.args = {}

    solve_problem._solve_pdm(print_output=True)

    assert solve_problem.above.optimised is True
    assert solve_problem.below.optimised is True
    assert solve_problem.case.post_processed is True
    assert solve_problem.case.output_called is True


def test_internal_problem_stage_reduction_transfers_nonisothermal_state_and_falls_back(
    monkeypatch,
):
    class ReducedCase:
        def __init__(self, **kwargs):
            self.S = kwargs["stages"]
            self.I = 1
            self.J = 1
            self.Q_r = [[[_Variable() for _k in range(self.S)] for _j in range(1)]]
            self.z = [[[_Variable() for _k in range(self.S)] for _j in range(1)]]
            self.theta_1 = [[[_Variable() for _k in range(self.S)] for _j in range(1)]]
            self.theta_2 = [[[_Variable() for _k in range(self.S)] for _j in range(1)]]
            self.X = [[[_Variable() for _k in range(self.S)] for _j in range(1)]]
            self.Y = [[[_Variable() for _k in range(self.S)] for _i in range(1)]]
            self.T_h_out_x = [
                [[_Variable() for _k in range(self.S)] for _j in range(1)]
            ]
            self.T_c_out_y = [
                [[_Variable() for _k in range(self.S)] for _i in range(1)]
            ]
            self.T_h = [[_Variable() for _k in range(self.S + 1)]]
            self.T_c = [[_Variable() for _k in range(self.S + 1)]]
            self.Q_c = [_Variable()]
            self.z_cu = [_Variable()]
            self.Q_h = [_Variable()]
            self.z_hu = [_Variable()]
            self.mSuccess = 0

        def optimise(self, print_output=False):
            self.optimise_called = True
            self.mSuccess = 0

    original_case = SimpleNamespace(
        mSuccess=1,
        I=1,
        J=1,
        S=2,
        Q_r=[[[[10.0], [0.0]]]],
        z=[[[[1.0], [0.0]]]],
        theta_1=[[[[2.0], [0.0]]]],
        theta_2=[[[[3.0], [0.0]]]],
        X=[[[[0.4], [0.0]]]],
        Y=[[[[0.5], [0.0]]]],
        T_h_out_x=[[[[120.0], [0.0]]]],
        T_c_out_y=[[[[80.0], [0.0]]]],
        T_h=[[[150.0], [120.0], [100.0]]],
        T_c=[[[60.0], [80.0], [90.0]]],
        Q_c=[[1.0]],
        z_cu=[[1.0]],
        Q_h=[[2.0]],
        z_hu=[[1.0]],
        non_isothermal_model=True,
        tol=1e-6,
        name="case",
        framework="TDM",
        solver_arrays=object(),
        dTmin=10.0,
        min_dqda=0.0,
        minimisation_goal="total utility",
        TAC_model="model",
        TAC=1.0,
        solve_time=0.1,
    )
    monkeypatch.setattr(unit_problem, "StageWiseModel", ReducedCase)
    problem = InternalHeatExchangerNetworkProblem(
        solver_arrays=object(),
        framework="TDM",
        stages=2,
    )

    failed_case = SimpleNamespace(mSuccess=0)
    assert problem.remove_unused_stages(failed_case) is failed_case
    assert problem.remove_unused_stages(original_case) is original_case


def test_stagewise_recovery_binary_and_scalar_value_fallbacks():
    z_values = [[[0]]]
    stagewise.StageWiseModel._set_recovery_binary_value(None, z_values, (0, 0, 0), 1)
    assert z_values == [[[1]]]

    z_values = [[[_IndexRejectingElement()]]]
    stagewise.StageWiseModel._set_recovery_binary_value(None, z_values, (0, 0, 0), 1)
    assert z_values == [[[1]]]

    assert stagewise_verification._value(7.5) == pytest.approx(7.5)
    assert stagewise_verification._value(
        SimpleNamespace(VALUE=SimpleNamespace(value=8.5))
    ) == pytest.approx(8.5)
    assert stagewise_verification._value(
        SimpleNamespace(value=SimpleNamespace(value=[9.5]))
    ) == pytest.approx(9.5)

    class FloatOnly:
        def __float__(self):
            return 10.5

    assert stagewise_verification._value(FloatOnly()) == pytest.approx(10.5)


def test_stagewise_temperature_checks_report_infeasible_active_matches():
    inactive_case = SimpleNamespace(
        I=1,
        J=1,
        S=1,
        Q_r=[[[[0.0]]]],
        non_isothermal_model=False,
        T_h=[[[120.0], [100.0]]],
        T_c=[[[80.0], [60.0]]],
    )
    assert stagewise_verification._check_temperatures(
        inactive_case,
        rel_tol=1e-6,
        abs_tol=1e-6,
        q_tol=1e-2,
    )

    crossing_case = SimpleNamespace(
        I=1,
        J=1,
        S=1,
        Q_r=[[[[5.0]]]],
        non_isothermal_model=False,
        T_h=[[[70.0], [60.0]]],
        T_c=[[[80.0], [65.0]]],
    )
    assert (
        stagewise_verification._check_temperatures(
            crossing_case,
            rel_tol=1e-6,
            abs_tol=1e-6,
            q_tol=1e-2,
        )
        is False
    )

    state_nonisothermal = SimpleNamespace(
        N_periods=1,
        I=1,
        J=1,
        S=1,
        Q_r_by_period=[[[[5.0]]]],
        non_isothermal_model=True,
        T_h_by_period=[[[90.0, 60.0]]],
        T_c_by_period=[[[50.0, 70.0]]],
        T_c_out_y_by_period=[[[[95.0]]]],
        theta_1_by_period=[[[[1.0]]]],
        T_h_out_x_by_period=[[[[60.0]]]],
        theta_2_by_period=[[[[1.0]]]],
    )
    assert (
        stagewise_verification._check_state_temperatures(
            state_nonisothermal,
            rel_tol=1e-6,
            abs_tol=1e-6,
            q_tol=1e-2,
        )
        is False
    )

    state_isothermal = SimpleNamespace(
        N_periods=1,
        I=1,
        J=1,
        S=1,
        Q_r_by_period=[[[[5.0]]]],
        non_isothermal_model=False,
        T_h_by_period=[[[70.0, 60.0]]],
        T_c_by_period=[[[80.0, 65.0]]],
    )
    assert (
        stagewise_verification._check_state_temperatures(
            state_isothermal,
            rel_tol=1e-6,
            abs_tol=1e-6,
            q_tol=1e-2,
        )
        is False
    )


def test_stagewise_shared_area_cost_checks_each_failure_path():
    base = dict(
        N_periods=1,
        I=1,
        J=1,
        S=1,
        U_r_period=[[[1.0]]],
        theta_1_by_period=[[[[1.0]]]],
        theta_2_by_period=[[[[1.0]]]],
        U_hu_period=[[1.0]],
        T_hu_in_period=[[200.0]],
        T_hu_out_period=[[180.0]],
        T_c_out_period=[[100.0]],
        T_c_by_period=[[[80.0, 90.0]]],
        U_cu_period=[[1.0]],
        T_h_by_period=[[[150.0, 120.0]]],
        T_h_out_period=[[110.0]],
        T_cu_out_period=[[40.0]],
        T_cu_in_period=[[30.0]],
        A_coeff=[1.0],
        A_exp=[1.0],
        hu_coeff=[1.0],
        hu_exp=[1.0],
        cu_coeff=[1.0],
        cu_exp=[1.0],
        recovery_area_cost_total=0.0,
        hu_area_cost_total=0.0,
        cu_area_cost_total=0.0,
        _utility_solved_outlet_temperature=(
            lambda side, period_idx, match_index, heat_duty: (
                180.0 if side == "hot" else 40.0
            )
        ),
    )

    no_recovery = SimpleNamespace(
        **base,
        area_r_shared=[[[0.0]]],
        Q_r_by_period=[[[[0.0]]]],
        area_hu_shared=[0.0],
        Q_h_by_period=[[0.0]],
        area_cu_shared=[0.0],
        Q_c_by_period=[[0.0]],
    )
    assert stagewise_verification._check_area_costs(no_recovery) is True

    bad_recovery = SimpleNamespace(
        **base,
        area_r_shared=[[[0.0]]],
        Q_r_by_period=[[[[5.0]]]],
        area_hu_shared=[100.0],
        Q_h_by_period=[[0.0]],
        area_cu_shared=[100.0],
        Q_c_by_period=[[0.0]],
    )
    assert stagewise_verification._check_area_costs(bad_recovery) is False

    bad_hot_utility = SimpleNamespace(
        **base,
        area_r_shared=[[[100.0]]],
        Q_r_by_period=[[[[0.0]]]],
        area_hu_shared=[0.0],
        Q_h_by_period=[[10_000.0]],
        area_cu_shared=[100.0],
        Q_c_by_period=[[0.0]],
    )
    assert stagewise_verification._check_area_costs(bad_hot_utility) is False

    bad_cold_utility = SimpleNamespace(
        **base,
        area_r_shared=[[[100.0]]],
        Q_r_by_period=[[[[0.0]]]],
        area_hu_shared=[100.0],
        Q_h_by_period=[[0.0]],
        area_cu_shared=[0.0],
        Q_c_by_period=[[10_000.0]],
    )
    assert stagewise_verification._check_area_costs(bad_cold_utility) is False
