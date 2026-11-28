"""A target's JSON is ``problem.results``, not the target itself.

Domain targets hold problem tables, stream collections, the configuration
and solver details, none of them JSON; ``problem.results`` is the form that
serialises them. ``target.model_dump(mode="json")`` raised a serialisation
error on the configuration (review finding 37; decided 2026-11-28 that
``problem.results`` is the route).
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem


def test_a_targets_json_dump_names_the_results_route():
    target = PinchProblem("basic_pinch.json").target.direct_heat_integration()

    with pytest.raises(TypeError, match="problem.results"):
        target.model_dump(mode="json")


def test_a_targets_json_string_names_the_results_route():
    target = PinchProblem("basic_pinch.json").target.direct_heat_integration()

    with pytest.raises(TypeError, match="problem.results"):
        target.model_dump_json()
