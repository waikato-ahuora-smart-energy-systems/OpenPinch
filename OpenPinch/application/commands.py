"""JSON-in/JSON-out engine commands shared by the browser and backend engines."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, get_type_hints
from uuid import uuid4

from pydantic import BaseModel

from ..adapters.io.problem_sources import load_packaged_sample_case
from ..contracts.reporting import TargetResults
from ..domain.targets import BaseTargetModel
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
    if "data" in source:
        if not isinstance(source["data"], Mapping):
            raise ValueError("Problem 'data' must be a JSON object.")
        return PinchProblem(source["data"])
    # Packaged cases only: a path here would read the engine host's files.
    return PinchProblem(load_packaged_sample_case(source["sample_case"]))


def run_command(request: Mapping[str, Any]) -> dict[str, Any]:
    """Run one engine command on a freshly loaded problem or a session's problem.

    ``request`` has ``command`` (``"target.<method>"``, ``"design.<method>"``
    or a deeper path such as ``"target.all_periods.<method>"``), either
    ``problem`` (``{"data": <TargetInput JSON>}`` or ``{"sample_case": <name>}``)
    or ``session``, and optional ``arguments`` for the method. ``"open"`` loads
    ``problem`` into a new session and returns its id as ``session``;
    ``"close"`` discards ``session``.

    The response has ``value``, the contract the method returned, and
    ``results``, the problem's ``TargetOutput`` (``None`` when no target has
    run), both as JSON-ready data. A returned case is ``{"data": ...}``, and
    in a session it also opens as a new ``session``. In a session a returned
    target is ``{"ref": <id>, "target": <TargetResults JSON>}``, and later
    arguments pass it as ``{"$ref": <id>}``; outside a session targets are
    reported only in the results.
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
    namespace, _, path = command.partition(".")
    method = {"target": problem.target, "design": problem.design}.get(namespace)
    for name in path.split("."):
        if name.startswith("_"):
            raise ValueError(f"Unknown command {command!r}.")
        method = getattr(method, name, None)
    if not callable(method):
        raise ValueError(f"Unknown command {command!r}.")
    contracts = _contract_parameters(method)
    arguments = {}
    for name, argument in request.get("arguments", {}).items():
        argument = _resolve(argument, session)
        if name in contracts and isinstance(argument, Mapping):
            argument = contracts[name].model_validate(argument)
        arguments[name] = argument
    returned = method(**arguments)
    if isinstance(returned, PinchProblem):
        # A new case, such as the optimised case from utility placement.
        value = {"data": returned.to_problem_json()}
        if session is not None:
            value["session"] = _open(returned)
    elif isinstance(returned, BaseTargetModel) and session is not None:
        ref = uuid4().hex
        session.targets[ref] = returned
        # By value too: a base_target command solves a detached residual
        # case, so its target is not in this problem's results.
        value = {"ref": ref, "target": _target_json(returned)}
    else:
        value = _to_json(returned)
    return {"value": value, "results": _to_json(problem.results)}


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
