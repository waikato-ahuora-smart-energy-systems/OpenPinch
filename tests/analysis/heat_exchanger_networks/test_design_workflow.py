"""HENS-09 public service, closed-contract, and documentation examples."""

from __future__ import annotations

import importlib.util
import json
import tomllib
from copy import deepcopy

import pytest
from pydantic import ValidationError

import OpenPinch
import OpenPinch.analysis.heat_exchanger_networks as synthesis_package
import OpenPinch.analysis.heat_exchanger_networks.targeting.open_hens_method as workflow_module
from OpenPinch.analysis.heat_exchanger_networks.execution.fake_executor import (
    FakeSynthesisExecutor,
)
from OpenPinch.analysis.heat_exchanger_networks.reporting.ranking import (
    network_structure_signature,
)
from OpenPinch.analysis.heat_exchanger_networks.service import (
    heat_exchanger_network_synthesis_service,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.application.workspace import PinchWorkspace
from OpenPinch.contracts.input import TargetInput
from OpenPinch.contracts.synthesis.common import HeatExchangerNetworkSynthesisManifest
from OpenPinch.domain.enums import HeatExchangerKind
from OpenPinch.domain.heat_exchanger import HeatExchanger
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from tests.support.paths import FIXTURES_ROOT, REPOSITORY_ROOT

REPO_ROOT = REPOSITORY_ROOT
FOUR_STREAM_FIXTURE = (
    FIXTURES_ROOT / "openhens" / "Four-stream-Yee-and-Grossmann-1990-1.json"
)

FORBIDDEN_PUBLIC_NAMES = {
    "OpenHENS",
    "CaseStudy",
    "SynthesisStudy",
    "DesignSpace",
    "MethodSequence",
    "SolveSetup",
    "StudyOutputs",
    "run_synthesis_workflow",
    "run_heat_exchanger_network_synthesis",
    "heat_exchanger_network_synthesis_service",
    "HeatExchangerNetworkDesignSpace",
    "HeatExchangerNetworkMethodSequence",
    "HeatExchangerNetworkSolveSetup",
    "HeatExchangerNetworkOutputs",
}


class _IterableDesignOptions:
    def __iter__(self):
        return iter((("period_id", "0"),))


def test_hen_public_exports_match_intended_snapshot() -> None:
    assert OpenPinch.__all__ == ["PinchProblem", "PinchWorkspace"]
    assert not hasattr(synthesis_package, "__all__")
    assert HeatExchanger.__module__ == "OpenPinch.domain.heat_exchanger"
    assert HeatExchangerNetwork.__module__ == "OpenPinch.domain.heat_exchanger_network"
    assert (
        HeatExchangerNetworkSynthesisManifest.__module__
        == "OpenPinch.contracts.synthesis.common"
    )


def test_no_openhens_compatibility_surface_or_import_shim() -> None:
    modules = [OpenPinch, synthesis_package]

    for module in modules:
        public_names = set(getattr(module, "__all__", ()))
        assert FORBIDDEN_PUBLIC_NAMES.isdisjoint(public_names)
        for name in FORBIDDEN_PUBLIC_NAMES:
            assert not hasattr(module, name)

    assert importlib.util.find_spec("OpenPinch.openhens") is None
    assert importlib.util.find_spec("OpenPinch.OpenHENS") is None


def test_no_openhens_command_parity_contract() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    scripts = pyproject["project"]["scripts"]

    assert scripts == {"openpinch": "OpenPinch.__main__:main"}
    assert "openhens" not in scripts


@pytest.mark.parametrize(
    "raw_input",
    [
        [{"section": "streams", "name": "raw csv row"}],
        [{"name": "H1", "t_supply": 400.0}],
        [{"name": "hot utility", "type": "Hot"}],
        TargetInput(streams=[], utilities=[]),
        type("CaseStudy", (), {})(),
        type("SynthesisStudy", (), {})(),
    ],
)
def test_internal_service_requires_live_problem(raw_input: object) -> None:
    with pytest.raises(TypeError, match="live PinchProblem"):
        heat_exchanger_network_synthesis_service(raw_input)  # type: ignore[arg-type]


def test_design_options_are_validated_at_their_owner_boundary(monkeypatch) -> None:
    _use_fake_default_executor(monkeypatch)
    problem = _public_example_problem()

    with pytest.raises(TypeError, match="runtime options.*dict"):
        heat_exchanger_network_synthesis_service(
            problem,
            options=TargetInput(streams=[], utilities=[]),  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="TargetInput.options"):
        heat_exchanger_network_synthesis_service(
            problem,
            options={"HENS_APPROACH_TEMPERATURES": [99.0]},
        )

    configured = problem.master_zone.config.hens.derivative_thresholds
    view = problem.design.heat_exchanger_network(
        options={"HENS_DERIVATIVE_THRESHOLDS": [9.9]},
    )
    assert view.result.manifest is not None
    assert problem.master_zone.config.hens.derivative_thresholds == configured


@pytest.mark.parametrize(
    "options",
    [
        TargetInput(streams=[], utilities=[]),
        _IterableDesignOptions(),
        type("CaseStudy", (), {})(),
        type("SynthesisStudy", (), {})(),
    ],
)
def test_public_design_accessor_rejects_separate_options_objects(
    options: object,
) -> None:
    problem = _public_example_problem()

    with pytest.raises(TypeError, match="design options.*mapping"):
        problem.design.heat_exchanger_network(options=options)  # type: ignore[arg-type]


def test_public_design_accessor_separates_runtime_and_configuration() -> None:
    problem = _public_example_problem()

    runtime_options, configuration = problem.design._arguments(
        options={"existing": True, "HENS_STAGE_SELECTION": [2]},
        period_id="peak",
    )

    assert runtime_options == {"existing": True, "period_id": "peak"}
    assert configuration == {"HENS_STAGE_SELECTION": [2]}


def test_openhens_field_aliases_are_rejected_by_public_schemas() -> None:
    with pytest.raises(ValidationError):
        HeatExchangerNetworkSynthesisManifest(
            run_id="example-run",
            approach_temperatures=(2.0,),
            derivative_thresholds=(0.5,),
            stage_selection=(2,),
            min_dT_values=(2.0,),
        )

    with pytest.raises(ValidationError):
        TargetInput(streams=[], options={"min_dT_values": [2.0]})


def test_problem_design_workflow_example_from_converted_openhens_fixture(
    monkeypatch,
) -> None:
    _use_fake_default_executor(monkeypatch)
    problem = _public_example_problem()

    design = problem.design.heat_exchanger_network()
    assert problem.results is not None
    assert problem.results.design == design.result
    assert design.result.manifest is not None
    assert design.result.manifest.run_id == "Four-stream-Yee-and-Grossmann-1990-1"
    assert isinstance(design.selected_network, HeatExchangerNetwork)
    assert design.selected_network.exchangers
    first_link = design.selected_network.exchangers[0]
    assert isinstance(first_link, HeatExchanger)
    assert first_link.source_stream
    assert first_link.sink_stream


def test_enhanced_synthesis_method_is_public_quality_tier_entrypoint(
    monkeypatch,
) -> None:
    _use_fake_default_executor(monkeypatch)
    problem = _public_example_problem()
    problem.update_options({"HENS_SYNTHESIS_QUALITY_TIER": 5})
    configured_tier = problem.master_zone.config.hens.synthesis_quality_tier

    design = problem.design.enhanced_heat_exchanger_network(quality_tier=3)

    assert problem.results is not None
    assert problem.results.design == design.result
    assert design.result.manifest is not None
    assert design.result.manifest.synthesis_quality_tier == 3
    assert problem.master_zone.config.hens.synthesis_quality_tier == configured_tier


def test_problem_design_view_is_created_only_by_explicit_design() -> None:
    problem = _public_example_problem()

    assert problem.results is None
    assert not hasattr(problem.design, "network")


def test_public_design_accessor_returns_ranked_networks(
    monkeypatch,
) -> None:
    class DuplicateStructureExecutor(FakeSynthesisExecutor):
        def execute(self, tasks, *, problem, parent_outcomes, max_parallel):
            outcomes = super().execute(
                tasks,
                problem=problem,
                parent_outcomes=parent_outcomes,
                max_parallel=max_parallel,
            )
            if not tasks or tasks[0].method != "network_evolution_method":
                return outcomes

            adjusted = []
            for outcome in outcomes:
                network = outcome.network
                if network is None or outcome.task.approach_temperature != 4.0:
                    adjusted.append(outcome)
                    continue

                exchangers = tuple(
                    (
                        exchanger.model_copy(update={"stage": 1})
                        if exchanger.kind is HeatExchangerKind.RECOVERY
                        else exchanger
                    )
                    for exchanger in network.exchangers
                )
                adjusted.append(
                    outcome.model_copy(
                        update={
                            "network": network.model_copy(
                                update={"exchangers": exchangers}
                            )
                        }
                    )
                )
            return tuple(adjusted)

    monkeypatch.setattr(
        workflow_module,
        "LocalSynthesisExecutor",
        DuplicateStructureExecutor,
    )
    payload = _public_example_payload()
    payload["options"] = {
        **payload["options"],
        "HENS_APPROACH_TEMPERATURES": [2.0, 4.0],
        "HENS_DERIVATIVE_THRESHOLDS": [0.5, 1.0],
        "HENS_STAGE_SELECTION": [2],
    }
    problem = PinchProblem(
        source=payload,
        project_name="Ranked unique heat exchanger network outcomes",
    )

    design = problem.design.heat_exchanger_network()
    serialized_before = design.result.model_dump(mode="json")

    assert not hasattr(design, "model_dump")
    assert not hasattr(design, "ranked_networks")
    assert not hasattr(design, "manifest")

    assert problem.results is not None
    assert problem.results.design == design.result
    assert len(design.result.ranked_networks) == 1
    assert design.top(1) == design.result.ranked_networks[:1]
    assert design.network(rank=1) == design.selected_network
    with pytest.raises(ValueError, match="one-based"):
        design.network(rank=0)
    with pytest.raises(IndexError):
        design.network(rank=2)
    assert design.result.ranked_networks[0].network == design.selected_network
    assert design.result.task_id == design.result.ranked_networks[0].task.task_id
    assert [
        outcome.objective_value for outcome in design.result.ranked_networks
    ] == sorted(outcome.objective_value for outcome in design.result.ranked_networks)
    assert all(outcome.status == "success" for outcome in design.result.ranked_networks)
    assert all(
        outcome.task.method == design.result.method
        for outcome in design.result.ranked_networks
    )
    assert len(
        {
            network_structure_signature(outcome.network)
            for outcome in design.result.ranked_networks
            if outcome.network is not None
        }
    ) == len(design.result.ranked_networks)

    selected_network = design.selected_network
    helper = design
    hot_utility = next(
        exchanger
        for exchanger in selected_network.exchangers
        if exchanger.kind is HeatExchangerKind.HOT_UTILITY
    )
    cold_utility = next(
        exchanger
        for exchanger in selected_network.exchangers
        if exchanger.kind is HeatExchangerKind.COLD_UTILITY
    )
    assert helper.total_heat_recovery == selected_network.total_duty(
        kind=HeatExchangerKind.RECOVERY
    )
    assert helper.total_hot_utility == selected_network.total_duty(
        kind=HeatExchangerKind.HOT_UTILITY
    )
    assert helper.total_cold_utility == selected_network.total_duty(
        kind=HeatExchangerKind.COLD_UTILITY
    )
    assert helper.utility(hot_utility.source_stream) == hot_utility.state().duty
    assert helper.utility(cold_utility.sink_stream) == cold_utility.state().duty
    assert helper.utility(hot_utility.sink_stream) == 0.0
    with pytest.raises(ValueError, match="utility name"):
        helper.utility("")
    assert design.result.model_dump(mode="json") == serialized_before


def test_native_targetinput_design_workflow_example_from_converted_fixture(
    monkeypatch,
) -> None:
    _use_fake_default_executor(monkeypatch)
    target_input = TargetInput.model_validate(_public_example_payload())
    problem = PinchProblem(
        source=target_input,
        project_name="Four-stream converted OpenHENS native example",
    )

    design = problem.design.heat_exchanger_network()

    assert problem.results is not None
    assert problem.results.design == design.result
    assert design.selected_network.exchangers


def test_workspace_design_workflow_example_from_converted_openhens_fixture(
    monkeypatch,
) -> None:
    _use_fake_default_executor(monkeypatch)
    workspace = PinchWorkspace(
        _public_example_payload(),
        project_name="Four-stream converted OpenHENS example",
    )

    design = workspace.design.heat_exchanger_network()
    problem = workspace.case("baseline")

    assert design.selected_network.exchangers
    assert problem.results is not None
    assert problem.results.design is not None
    assert problem.results.design.network.exchangers


def _public_example_problem() -> PinchProblem:
    return PinchProblem(
        source=_public_example_payload(),
        project_name="Four-stream converted OpenHENS example",
    )


def _use_fake_default_executor(monkeypatch) -> None:
    from OpenPinch.analysis.heat_exchanger_networks.targeting import (
        network_evolution_method,
        pinch_design_method,
        thermal_derivative_method,
    )

    for module in (
        workflow_module,
        network_evolution_method,
        pinch_design_method,
        thermal_derivative_method,
    ):
        monkeypatch.setattr(module, "LocalSynthesisExecutor", FakeSynthesisExecutor)


def _public_example_payload() -> dict:
    payload = json.loads(FOUR_STREAM_FIXTURE.read_text(encoding="utf-8"))
    return deepcopy(payload)
