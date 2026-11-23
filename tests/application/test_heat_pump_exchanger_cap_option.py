"""``HENS_HEAT_PUMP_STREAM_EXCHANGERS``: exchangers one heat pump stream may have.

A condenser or an evaporator is one piece of equipment, or a few in
series or parallel on the working fluid. A network that serves it through
many exchangers is not a machine anyone would build, so a design keeps
only the networks within this number, two by default.
"""

from __future__ import annotations

import pytest

from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.configuration_fields import USER_CONFIG_FIELD_SPECS


def test_the_exchanger_cap_is_a_configuration_key_of_at_least_one():
    spec = USER_CONFIG_FIELD_SPECS["HENS_HEAT_PUMP_STREAM_EXCHANGERS"]

    assert spec.default == 2
    assert spec.config_path == ("hens", "heat_pump_stream_exchangers")
    assert Configuration().hens.heat_pump_stream_exchangers == 2
    configured = Configuration(options={"HENS_HEAT_PUMP_STREAM_EXCHANGERS": 1})
    assert configured.hens.heat_pump_stream_exchangers == 1
    with pytest.raises(ValueError, match="HENS_HEAT_PUMP_STREAM_EXCHANGERS"):
        Configuration(options={"HENS_HEAT_PUMP_STREAM_EXCHANGERS": 0})
