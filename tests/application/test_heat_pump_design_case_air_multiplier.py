"""An air-source heat pump is designed at any dt_cont multiplier.

The target stands the evaporator the air's own contribution
(``HPR_DT_ENV_CONT``) under the air, whatever the multiplier; the design
scales every stream's contribution by the multiplier. The design case
wrote the air at its own contribution, so at a multiplier of 1.5 the
design held the evaporator 15 K under the air where the target had put it
10 K under, and the case targeted 405 kW of utility the heat pump had
left at none.
"""

from __future__ import annotations

import pytest

from OpenPinch.application.heat_pump_design import heat_pump_design_case
from tests.application.test_heat_pump_stream_case_air_source import _short_of_heat


@pytest.mark.parametrize("multiplier", [0.75, 1.5])
def test_the_air_keeps_its_own_contribution_at_the_designs_multiplier(multiplier):
    run = heat_pump_design_case(
        _short_of_heat(),
        multiplier,
        load={"fraction": 1.0},
        levels={"condensers": 1, "evaporators": 1},
        # The search's first start, which draws on air (see
        # heat_pump_stream_cases.full_load_target).
        search={"restarts": 0, "evaluations": 1},
    )

    (air,) = [
        record
        for record in run.case.to_problem_json()["streams"]
        if (record.get("heat_pump") or {}).get("role") == "ambient_source"
    ]
    # HPR_DT_ENV_CONT is 10 K: what the design's multiplier must scale to.
    assert air["dt_cont"]["value"] * multiplier == pytest.approx(10.0)
