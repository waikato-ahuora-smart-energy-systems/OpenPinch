"""HENS-09 public service, compatibility, and documentation examples."""

from __future__ import annotations

import importlib.util
import json
import tomllib
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

import OpenPinch
import OpenPinch.classes
import OpenPinch.lib
import OpenPinch.lib.schemas as schemas
import OpenPinch.services
import OpenPinch.services.heat_exchanger_network_synthesis as synthesis_package
import OpenPinch.services.heat_exchanger_network_synthesis.targeting_services.open_hens_method as workflow_module
from OpenPinch import PinchProblem, PinchWorkspace
from OpenPinch.classes.heat_exchanger import HeatExchanger
from OpenPinch.classes.heat_exchanger_network import HeatExchangerNetwork
from OpenPinch.lib import HeatExchangerKind
from OpenPinch.lib.schemas.io import TargetInput
from OpenPinch.lib.schemas.synthesis import HeatExchangerNetworkSynthesisManifest
from OpenPinch.services.heat_exchanger_network_synthesis.common.execution.fake_executor import (
    FakeSynthesisExecutor,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.reporting.ranking import (
    network_structure_signature,
)
from OpenPinch.services.heat_exchanger_network_synthesis.heat_exchanger_network_synthesis_entry import (
    heat_exchanger_network_synthesis_service,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FOUR_STREAM_FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "openhens"
    / "Four-stream-Yee-and-Grossmann-1990-1.json"
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

HEN_PUBLIC_NAME_SNAPSHOT = {
    "OpenPinch": set(),
    "OpenPinch.classes": {
        "HeatExchanger",
        "HeatExchangerKind",
        "HeatExchangerNetwork",
        "HeatExchangerStreamRole",
    },
    "OpenPinch.lib": {
        "HEN",
        "HeatExchangerKind",
        "HeatExchangerNetworkLabel",
        "HeatExchangerNetworkLabelKey",
        "HeatExchangerNetworkSynthesisExportRecord",
        "HeatExchangerNetworkSynthesisManifest",
        "HeatExchangerNetworkSynthesisMethodInput",
        "HeatExchangerNetworkSynthesisMethodOutput",
        "HeatExchangerNetworkSynthesisResult",
        "HeatExchangerNetworkSynthesisTask",
        "HeatExchangerNetworkSynthesisTaskOutcome",
        "HeatExchangerNetworkTopologyRestriction",
        "HeatExchangerStreamRole",
        "SynthesisMethod",
        "SynthesisOutputFormat",
        "SynthesisTaskStatus",
    },
    "OpenPinch.lib.schemas": {
        "HeatExchangerNetworkSynthesisExportRecord",
        "HeatExchangerNetworkSynthesisManifest",
        "HeatExchangerNetworkSynthesisMethodInput",
        "HeatExchangerNetworkSynthesisMethodOutput",
        "HeatExchangerNetworkSynthesisResult",
        "HeatExchangerNetworkSynthesisTask",
        "HeatExchangerNetworkSynthesisTaskOutcome",
        "HeatExchangerNetworkTopologyRestriction",
        "SynthesisMethod",
        "SynthesisOutputFormat",
        "SynthesisTaskStatus",
    },
    "OpenPinch.services": set(),
    "OpenPinch.services.heat_exchanger_network_synthesis": set(),
}
HEN_SNAPSHOT_NAMES = set().union(*HEN_PUBLIC_NAME_SNAPSHOT.values())


class _IterableDesignOptions:
    def __iter__(self):
        return iter((("period_id", "0"),))


def test_hen_public_exports_match_intended_snapshot() -> None:
    modules = {
        "OpenPinch": OpenPinch,
        "OpenPinch.classes": OpenPinch.classes,
        "OpenPinch.lib": OpenPinch.lib,
        "OpenPinch.lib.schemas": schemas,
        "OpenPinch.services": OpenPinch.services,
        "OpenPinch.services.heat_exchanger_network_synthesis": synthesis_package,
    }

    for module_name, module in modules.items():
        assert (
            _hen_related_public_names(module) == HEN_PUBLIC_NAME_SNAPSHOT[module_name]
        )


def test_no_openhens_compatibility_surface_or_import_shim() -> None:
    modules = [
        OpenPinch,
        OpenPinch.classes,
        OpenPinch.lib,
        schemas,
        OpenPinch.services,
        synthesis_package,
    ]

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
def test_internal_service_requires_live_pinch_problem(raw_input: object) -> None:
    with pytest.raises(TypeError, match="live PinchProblem"):
        heat_exchanger_network_synthesis_service(raw_input)  # type: ignore[arg-type]


def test_internal_service_rejects_separate_design_options() -> None:
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

    with pytest.raises(ValueError, match="TargetInput.options"):
        problem.design.heat_exchanger_network_synthesis(
            options={"HENS_DERIVATIVE_THRESHOLDS": [9.9]},
        )


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

    with pytest.raises(TypeError, match="runtime options.*dict"):
        problem.design.heat_exchanger_network_synthesis(options=options)  # type: ignore[arg-type]


def test_public_design_accessor_runtime_options_include_period_id() -> None:
    problem = _public_example_problem()

    runtime_options = problem.design._runtime_options(
        {"existing": True},
        period_id="peak",
    )

    assert runtime_options == {"existing": True, "period_id": "peak"}


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

    design = problem.design.heat_exchanger_network_synthesis()

    assert problem.results is not None
    assert problem.results.design == design
    assert design.manifest is not None
    assert design.manifest.run_id == "Four-stream-Yee-and-Grossmann-1990-1"
    assert isinstance(design.network, HeatExchangerNetwork)
    assert design.network.exchangers
    first_link = design.network.exchangers[0]
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

    design = problem.design.enhanced_synthesis_method(quality_tier=3)

    assert problem.results is not None
    assert problem.results.design == design
    assert design.manifest is not None
    assert design.manifest.synthesis_quality_tier == 3
    assert problem.master_zone.config.hens.synthesis_quality_tier == configured_tier


def test_problem_design_network_helpers_require_cached_design() -> None:
    problem = _public_example_problem()

    with pytest.raises(RuntimeError, match="heat exchanger network design method"):
        _ = problem.design.network.total_heat_recovery


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

    design = problem.design.heat_exchanger_network_synthesis()

    assert problem.results is not None
    assert problem.results.design == design
    assert len(design.ranked_networks) == 1
    assert design.ranked_networks[0].network == design.network
    assert design.task_id == design.ranked_networks[0].task.task_id
    assert [outcome.objective_value for outcome in design.ranked_networks] == sorted(
        outcome.objective_value for outcome in design.ranked_networks
    )
    assert all(outcome.status == "success" for outcome in design.ranked_networks)
    assert all(
        outcome.task.method == design.method for outcome in design.ranked_networks
    )
    assert len(
        {
            network_structure_signature(outcome.network)
            for outcome in design.ranked_networks
            if outcome.network is not None
        }
    ) == len(design.ranked_networks)

    selected_network = design.network
    helper = problem.design.network
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
    assert helper.utility(hot_utility.source_stream) == hot_utility.duty
    assert helper.utility(cold_utility.sink_stream) == cold_utility.duty
    assert helper.utility(hot_utility.sink_stream) == 0.0
    with pytest.raises(ValueError, match="utility name"):
        helper.utility("")


def test_native_targetinput_design_workflow_example_from_converted_fixture(
    monkeypatch,
) -> None:
    _use_fake_default_executor(monkeypatch)
    target_input = TargetInput.model_validate(_public_example_payload())
    problem = PinchProblem(
        source=target_input,
        project_name="Four-stream converted OpenHENS native example",
    )

    design = problem.design.heat_exchanger_network_synthesis()

    assert problem.results is not None
    assert problem.results.design == design
    assert design.network.exchangers


def test_workspace_design_workflow_example_from_converted_openhens_fixture(
    monkeypatch,
) -> None:
    _use_fake_default_executor(monkeypatch)
    workspace = PinchWorkspace(
        _public_example_payload(),
        project_name="Four-stream converted OpenHENS example",
    )

    view = workspace.solve_variant(
        "baseline",
        workflow="heat_exchanger_network_synthesis",
    )
    problem = workspace.case("baseline")

    assert view.status == "solved"
    assert view.support_level == "advanced"
    assert problem.results is not None
    assert problem.results.design is not None
    assert problem.results.design.workspace_variant == "baseline"
    assert problem.results.design.network.exchangers


def _hen_related_public_names(module) -> set[str]:
    public_names = set(getattr(module, "__all__", ()))
    return {
        name
        for name in public_names
        if name in FORBIDDEN_PUBLIC_NAMES or name in HEN_SNAPSHOT_NAMES
    }


def _public_example_problem() -> PinchProblem:
    return PinchProblem(
        source=_public_example_payload(),
        project_name="Four-stream converted OpenHENS example",
    )


def _use_fake_default_executor(monkeypatch) -> None:
    from OpenPinch.services.heat_exchanger_network_synthesis.targeting_services import (
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
