"""``HENS_EVOLUTION_LEVEL_HOLDS``: whether network evolution of a single
process keeps the utility levels its pinch design held.

Pinch design with utility placement by temperature holds each utility level
at the duty targeting gives it. Evolution then trades utility for area by
price and leaves the targets, so only ranking kept a level-holding network.
The option asks evolution to keep them: "fixed" at each target, "none" (the
default) as before.
"""

from __future__ import annotations

import pytest

from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.configuration_fields import USER_CONFIG_FIELD_SPECS


def test_the_evolution_level_holds_are_a_configuration_key():
    spec = USER_CONFIG_FIELD_SPECS["HENS_EVOLUTION_LEVEL_HOLDS"]

    assert spec.default == "none"
    assert spec.config_path == ("hens", "evolution_level_holds")
    assert Configuration().hens.evolution_level_holds == "none"
    for holds in ("none", "fixed"):
        configured = Configuration(options={"HENS_EVOLUTION_LEVEL_HOLDS": holds})
        assert configured.hens.evolution_level_holds == holds
    # No "at_least", the site option's other value: measured on a single
    # process, a level held at or above its target does not bind and gives
    # the networks of "none".
    for unknown in ("at_least", "loose"):
        with pytest.raises(ValueError, match="HENS_EVOLUTION_LEVEL_HOLDS"):
            Configuration(options={"HENS_EVOLUTION_LEVEL_HOLDS": unknown})
