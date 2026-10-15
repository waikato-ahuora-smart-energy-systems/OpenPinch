"""JSON-in/JSON-out engine commands shared by the browser and backend engines."""

from __future__ import annotations

import json
from typing import Any, Mapping

from pydantic import BaseModel

from ..adapters.io.problem_sources import load_packaged_sample_case
from ..domain.targets import BaseTargetModel
from ._problem.accessors.design import HeatExchangerNetworkDesignView
from .problem import PinchProblem


def _load_problem(source: Mapping[str, Any]) -> PinchProblem:
    if "data" in source:
        if not isinstance(source["data"], Mapping):
            raise ValueError("Problem 'data' must be a JSON object.")
        return PinchProblem(source["data"])
    # Packaged cases only: a path here would read the engine host's files.
    return PinchProblem(load_packaged_sample_case(source["sample_case"]))


def run_command(request: Mapping[str, Any]) -> dict[str, Any]:
    """Run one engine command on a freshly loaded problem.

    ``request`` has ``command`` (``"target.<method>"`` or ``"design.<method>"``),
    ``problem`` (``{"data": <TargetInput JSON>}`` or ``{"sample_case": <name>}``)
    and optional ``arguments`` for the method. The response has ``value``, the
    contract the method returned (``None`` for targets, which are reported in
    the results), and ``results``, the problem's ``TargetOutput`` (``None``
    when no target has run), both as JSON-ready data.
    """
    command = request["command"]
    namespace, _, name = command.partition(".")
    problem = _load_problem(request["problem"])
    accessor = {"target": problem.target, "design": problem.design}.get(namespace)
    method = getattr(accessor, name, None)
    if name.startswith("_") or not callable(method):
        raise ValueError(f"Unknown command {command!r}.")
    returned = method(**request.get("arguments", {}))
    return {
        "value": _to_json(returned),
        "results": _to_json(problem.results),
    }


def run_command_json(request: str) -> str:
    """Run a command given and answered as JSON text (the Pyodide entry point)."""
    return json.dumps(run_command(json.loads(request)))


def _to_json(value: Any) -> Any:
    if isinstance(value, HeatExchangerNetworkDesignView):
        value = value.result
    # A target model is reported through the problem results, not by value.
    if isinstance(value, BaseModel) and not isinstance(value, BaseTargetModel):
        return value.model_dump(mode="json")
    return None
