"""A heat pump stream may not take the name of a stream the zone has.

Stream keys are the zone address and the name. A second stream of the
same name would be renamed on import, and the mark that names the heat
pump's streams would then point at a key nobody chose.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from tests.application.heat_pump_stream_cases import basic_source, full_load_target


def test_a_zone_that_has_a_stream_of_the_same_name_is_refused():
    source = basic_source()
    source["streams"][0]["name"] = "HP_H1"
    # With the tree given, the streams stay on the process zone itself.
    source["zone_tree"] = {
        "name": "Site",
        "type": "Site",
        "children": [{"name": "Plant", "type": "Process Zone"}],
    }
    problem = PinchProblem(source=source, project_name="basic")
    target = full_load_target(problem)

    with pytest.raises(ValueError, match="already has a stream named 'HP_H1'"):
        problem.with_heat_pump_streams(target)
