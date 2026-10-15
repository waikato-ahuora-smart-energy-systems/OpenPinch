"""JSON-in/JSON-out engine commands shared by the browser and backend engines."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Mapping, get_type_hints
from uuid import uuid4

from pydantic import BaseModel

from ..adapters.io.problem_sources import load_packaged_sample_case
from ..contracts.hpr import HPRTargetingError
from ..contracts.reporting import TargetResults
from ..domain.targets import BaseTargetModel
from ..presentation.reporting.hpr import hpr_failure_summary
from ..presentation.reporting.results import serialize_target
from ._problem.accessors.design import HeatExchangerNetworkDesignView
from .problem import PinchProblem


@dataclass
class _Session:
    """A live problem and the targets its commands returned, by reference id."""

    problem: PinchProblem
    targets: dict[str, BaseTargetModel] = field(default_factory=dict)


# Sessions kept between commands, by session id, for the life of the engine.
_SESSIONS: dict[str, _Session] = {}


def _load_problem(source: Mapping[str, Any]) -> PinchProblem:
    project_name = source.get("project_name", "Site")
    if "data" in source:
        if not isinstance(source["data"], Mapping):
            raise ValueError("Problem 'data' must be a JSON object.")
        return PinchProblem(source["data"], project_name=project_name)
    # Packaged cases only: a path here would read the engine host's files.
    return PinchProblem(
        load_packaged_sample_case(source["sample_case"]), project_name=project_name
    )


def run_command(request: Mapping[str, Any]) -> dict[str, Any]:
    """Run one engine command on a freshly loaded problem or a session's problem.

    ``request`` has ``command`` (``"target.<method>"``, ``"design.<method>"``,
    a deeper path such as ``"target.all_periods.<method>"``, or a problem
    command: ``validate``, ``to_problem_json``, ``streams``, ``report``,
    ``compare``, ``residual_utility``, ``with_utilities_from`` and
    ``components.add_process_mvr`` / ``inventory`` / ``activate`` /
    ``deactivate``), either ``problem`` (``{"data": <TargetInput JSON>}`` or
    ``{"sample_case": <name>}``, each with an optional ``project_name``) or
    ``session``, and optional ``arguments`` for the method. ``"open"`` loads
    ``problem`` into a new session and returns its id as ``session``;
    ``"close"`` discards ``session``.

    The response has ``value``, the contract the method returned, and
    ``results``, the problem's ``TargetOutput`` (``None`` when no target has
    run), both as JSON-ready data. A returned case is ``{"data": ...}``, with
    its ``results`` and ``utility_placement_result`` when it has them, and in
    a session it also opens as a new ``session``. In a session a returned
    target is ``{"ref": <id>, "target": <TargetResults JSON>}``, and later
    arguments pass it as ``{"$ref": <id>}``; another session's problem is
    passed as ``{"$session": <id>}``. Outside a session targets are reported
    only in the results. Analysis that runs and fails (heat pump targeting)
    returns ``error`` (``{type, message, details}``) instead of raising.
    """
    command = request["command"]
    if command == "open":
        session = _open(_load_problem(request["problem"]))
        return {"session": session, "value": None, "results": None}
    session = None
    if "session" in request:
        session = _SESSIONS.get(request["session"])
        if session is None:
            raise ValueError(f"Unknown session {request['session']!r}.")
        if command == "close":
            del _SESSIONS[request["session"]]
            return {"value": None, "results": None}
        problem = session.problem
    else:
        problem = _load_problem(request["problem"])
    handler = _PROBLEM_COMMANDS.get(command)
    if handler is not None:
        arguments = {
            name: _resolve(argument, session)
            for name, argument in request.get("arguments", {}).items()
        }
        method = partial(handler, problem)
    else:
        method = _accessor_method(problem, command)
        contracts = _contract_parameters(method)
        arguments = {}
        for name, argument in request.get("arguments", {}).items():
            argument = _resolve(argument, session)
            if name in contracts and isinstance(argument, Mapping):
                argument = contracts[name].model_validate(argument)
            arguments[name] = argument
    try:
        returned = method(**arguments)
    except _ANALYSIS_FAILURES as error:
        return {
            "value": None,
            "results": _to_json(problem.results),
            "error": _error_json(error),
        }
    if isinstance(returned, PinchProblem):
        # A new case, such as the optimised case from utility placement.
        value = {"data": returned.to_problem_json()}
        evidence = returned.utility_placement_result
        if evidence is not None:
            value["utility_placement_result"] = _to_json(evidence)
        if returned.results is not None:
            value["results"] = _to_json(returned.results)
        if session is not None:
            value["session"] = _open(returned)
    elif isinstance(returned, BaseTargetModel) and session is not None:
        ref = uuid4().hex
        session.targets[ref] = returned
        # By value too: a base_target command solves a detached residual
        # case, so its target is not in this problem's results.
        value = {"ref": ref, "target": _target_json(returned)}
    elif handler is not None:
        # Problem commands return JSON-ready data already.
        value = returned
    else:
        value = _to_json(returned)
    return {"value": value, "results": _to_json(problem.results)}


# Analysis that ran and failed is reported in the response; anything else
# (a bad request, a bug) raises.
_ANALYSIS_FAILURES = (HPRTargetingError,)


def _error_json(error: Exception) -> dict[str, Any]:
    details = (
        hpr_failure_summary(error) if isinstance(error, HPRTargetingError) else None
    )
    return {"type": type(error).__name__, "message": str(error), "details": details}


def _accessor_method(problem: PinchProblem, command: str) -> Any:
    """The public method a ``target.`` or ``design.`` command names."""
    namespace, _, path = command.partition(".")
    method = {"target": problem.target, "design": problem.design}.get(namespace)
    for name in path.split("."):
        if name.startswith("_"):
            raise ValueError(f"Unknown command {command!r}.")
        method = getattr(method, name, None)
    if not callable(method):
        raise ValueError(f"Unknown command {command!r}.")
    return method


def _validate(problem: PinchProblem) -> dict[str, Any]:
    return {
        "input": problem.validate().model_dump(mode="json"),
        "report": problem.validation_report().model_dump(mode="json"),
    }


def _streams(problem: PinchProblem) -> dict[str, Any]:
    """The prepared streams and utilities, one row per segment."""
    return {
        name: getattr(problem, name).to_dict(expand_segments=True)
        for name in ("hot_streams", "cold_streams", "hot_utilities", "cold_utilities")
    }


def _report(
    problem: PinchProblem,
    *,
    detailed: bool = False,
    include_periods: bool = False,
    include_weighted_average: bool = False,
) -> dict[str, Any]:
    """Metrics, the report and the summary table of the solved targets."""
    periods = {
        "include_periods": include_periods,
        "include_weighted_average": include_weighted_average,
    }
    summary = problem.summary_frame(detailed=detailed, **periods)
    return {
        "metrics": [
            metric.model_dump(mode="json") for metric in problem.metrics(**periods)
        ],
        "report": problem.report(**periods).model_dump(mode="json"),
        "summary": _frame_json(summary),
    }


def _component_json(component: Any) -> dict[str, Any]:
    """A process component as JSON, with streams by name."""
    return {
        "id": component.id,
        "component_type": component.component_type,
        "active": component.active,
        "affected_zone_paths": component.affected_zone_paths,
        "original_streams": [stream.name for stream in component.original_streams],
        "replacement_streams": [
            stream.name for stream in component.replacement_streams
        ],
    }


def _add_process_mvr(problem: PinchProblem, **arguments: Any) -> dict[str, Any]:
    return _component_json(problem.components.add_process_mvr(**arguments))


def _component(problem: PinchProblem, id: str) -> Any:
    component = problem.components.inventory.get(id)
    if component is None:
        raise ValueError(f"Unknown process component {id!r}.")
    return component


def _inventory(problem: PinchProblem) -> dict[str, Any]:
    return {
        id: _component_json(component)
        for id, component in problem.components.inventory.items()
    }


def _activate(problem: PinchProblem, *, id: str) -> dict[str, Any]:
    return _component_json(_component(problem, id).activate())


def _deactivate(problem: PinchProblem, *, id: str) -> dict[str, Any]:
    return _component_json(_component(problem, id).deactivate())


def _frame_json(frame: Any) -> dict[str, Any]:
    """A DataFrame as ``{columns, data}``; pandas writes NaN as null."""
    return json.loads(frame.to_json(orient="split", index=False))


def _compare(problem: PinchProblem, **arguments: Any) -> dict[str, Any]:
    return _frame_json(problem.compare_to(**arguments))


# Commands on the problem itself rather than on one of its accessors.
_PROBLEM_COMMANDS = {
    "validate": _validate,
    "residual_utility": PinchProblem.residual_utility,
    "with_utilities_from": PinchProblem.with_utilities_from,
    "compare": _compare,
    "components.add_process_mvr": _add_process_mvr,
    "components.inventory": _inventory,
    "components.activate": _activate,
    "components.deactivate": _deactivate,
    "report": _report,
    "streams": _streams,
    "to_problem_json": PinchProblem.to_problem_json,
}


def _target_json(target: BaseTargetModel) -> dict[str, Any]:
    """One target as the results table reports it (``TargetResults``)."""
    return TargetResults.model_validate(serialize_target(target)).model_dump(
        mode="json"
    )


def _open(problem: PinchProblem) -> str:
    session = uuid4().hex
    _SESSIONS[session] = _Session(problem)
    return session


def _contract_parameters(method: Any) -> dict[str, type[BaseModel]]:
    """Parameters typed as a contract, which JSON arguments are validated into."""
    try:
        hints = get_type_hints(method)
    except NameError:
        # Some hints name types imported only for type checking.
        return {}
    return {
        name: hint
        for name, hint in hints.items()
        if isinstance(hint, type)
        and issubclass(hint, BaseModel)
        and not issubclass(hint, BaseTargetModel)
    }


def _resolve(argument: Any, session: _Session | None) -> Any:
    if isinstance(argument, Mapping) and set(argument) == {"$session"}:
        # Another session's problem, for commands that take a second problem.
        other = _SESSIONS.get(argument["$session"])
        if other is None:
            raise ValueError(f"Unknown session {argument['$session']!r}.")
        return other.problem
    if not (isinstance(argument, Mapping) and set(argument) == {"$ref"}):
        return argument
    target = None if session is None else session.targets.get(argument["$ref"])
    if target is None:
        raise ValueError(f"Unknown target reference {argument['$ref']!r}.")
    return target


def run_command_json(request: str) -> str:
    """Run a command given and answered as JSON text (the Pyodide entry point)."""
    return json.dumps(run_command(json.loads(request)))


def _to_json(value: Any) -> Any:
    if isinstance(value, HeatExchangerNetworkDesignView):
        value = value.result
    if isinstance(value, Mapping):
        return {key: _to_json(item) for key, item in value.items()}
    # A target model is reported through the problem results, not by value.
    if isinstance(value, BaseModel) and not isinstance(value, BaseTargetModel):
        return value.model_dump(mode="json")
    return None
