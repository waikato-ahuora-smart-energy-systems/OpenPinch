"""The thermal derivative step prices utilities when a side has a choice.

Its objective was the hot utility duty. With one hot and one cold utility
that is the utility cost, since the cold duty follows the hot. With two cold
utilities it leaves the split between them free, and the solve put cooling a
cheap utility could do on a dear one.
"""

from __future__ import annotations

import copy
import json

from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
    LocalSynthesisExecutor,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.contracts.synthesis.task import HeatExchangerNetworkSynthesisTask
from tests.support.paths import FIXTURES_ROOT


def _four_stream_payload() -> dict:
    """Hot streams cool to 370 K against cooling water at 300 to 320 K."""
    path = FIXTURES_ROOT / "openhens" / "Four-stream-Yee-and-Grossmann-1990-1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["options"] = {
        key: value
        for key, value in payload["options"].items()
        if key != "HENS_OUTPUT_FOLDER"
    } | {"HENS_OUTPUT_FORMATS": []}
    return payload


def _two_cold_level_payload() -> dict:
    """The four-stream case with warm water (340 to 350 K) at a third of the
    cooling water price. It can do all the cooling."""
    payload = _four_stream_payload()
    cooling_water = next(u for u in payload["utilities"] if u["type"] == "Cold")
    warm_water = copy.deepcopy(cooling_water)
    warm_water["name"] = "Warm water"
    warm_water["price"]["value"] = cooling_water["price"]["value"] / 3.0
    warm_water["t_supply"]["value"] = 340.0
    warm_water["t_target"]["value"] = 350.0
    payload["utilities"].append(warm_water)
    return payload


def _thermal_derivative_goal(payload: dict) -> str:
    task = HeatExchangerNetworkSynthesisTask(
        run_id="method-test",
        method="thermal_derivative_method",
        dt_cont_multiplier=1.0,
        derivative_threshold=0.5,
        stage_count=1,
    )
    internal = LocalSynthesisExecutor(print_output=False)._build_problem(
        task,
        problem=PinchProblem(source=payload),
        parent_outcomes={},
    )
    return internal.minimisation_goal


def test_a_thermal_derivative_model_minimises_utility_cost_given_a_choice() -> None:
    assert _thermal_derivative_goal(_four_stream_payload()) == "hot utility"
    assert _thermal_derivative_goal(_two_cold_level_payload()) == "utility costs"
