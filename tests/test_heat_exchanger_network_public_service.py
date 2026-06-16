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
from OpenPinch import PinchProblem, PinchWorkspace
from OpenPinch.classes.heat_exchanger import HeatExchanger
from OpenPinch.classes.heat_exchanger_network import HeatExchangerNetwork
from OpenPinch.lib.schemas.io import TargetInput
from OpenPinch.lib.schemas.synthesis import HeatExchangerNetworkSynthesisManifest
from OpenPinch.services.heat_exchanger_network_synthesis.service import (
    heat_exchanger_network_synthesis_service,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FOUR_STREAM_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "openhens" / "Four-stream-Yee-and-Grossmann-1990-1.json"
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
        "HeatExchangerNetworkLabel",
        "HeatExchangerNetworkLabelKey",
        "HeatExchangerNetworkSynthesisExportRecord",
        "HeatExchangerNetworkSynthesisManifest",
        "HeatExchangerNetworkSynthesisResult",
        "HeatExchangerNetworkSynthesisTask",
        "HeatExchangerNetworkSynthesisTaskOutcome",
        "HeatExchangerNetworkTopologyRestriction",
        "SynthesisMethod",
        "SynthesisOutputFormat",
        "SynthesisTaskStatus",
    },
    "OpenPinch.lib.schemas": {
        "HeatExchangerNetworkSynthesisExportRecord",
        "HeatExchangerNetworkSynthesisManifest",
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
        return iter((("state_id", "0"),))


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
        assert _hen_related_public_names(module) == HEN_PUBLIC_NAME_SNAPSHOT[module_name]


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


def test_problem_design_workflow_example_from_converted_openhens_fixture() -> None:
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


def test_native_targetinput_design_workflow_example_from_converted_fixture() -> None:
    target_input = TargetInput.model_validate(_public_example_payload())
    problem = PinchProblem(
        source=target_input,
        project_name="Four-stream converted OpenHENS native example",
    )

    design = problem.design.heat_exchanger_network_synthesis()

    assert problem.results is not None
    assert problem.results.design == design
    assert design.network.exchangers


def test_workspace_design_workflow_example_from_converted_openhens_fixture() -> None:
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


def _public_example_payload() -> dict:
    payload = json.loads(FOUR_STREAM_FIXTURE.read_text(encoding="utf-8"))
    return deepcopy(payload)
