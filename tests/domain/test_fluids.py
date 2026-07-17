from __future__ import annotations

import pytest

from OpenPinch.domain.fluids import (
    build_coolprop_abstract_state,
    validate_coolprop_fluid_name,
)


def test_build_coolprop_abstract_state_returns_existing_state_objects():
    state = object()

    assert build_coolprop_abstract_state(state) is state


def test_build_coolprop_abstract_state_validates_explicit_mixture_syntax():
    pytest.importorskip("CoolProp")

    with pytest.raises(ValueError, match="component.*mole_fraction"):
        build_coolprop_abstract_state("Water[0.5]&Ethanol")
    with pytest.raises(ValueError, match="finite and non-negative"):
        build_coolprop_abstract_state("Water[-0.5]&Ethanol[1.5]")
    with pytest.raises(ValueError, match="sum to a positive"):
        build_coolprop_abstract_state("Water[0.0]&Ethanol[0.0]")


def test_validate_coolprop_fluid_name_wraps_invalid_names():
    pytest.importorskip("CoolProp")

    with pytest.raises(ValueError, match="Invalid CoolProp fluid_name"):
        validate_coolprop_fluid_name("NotAFluid")
