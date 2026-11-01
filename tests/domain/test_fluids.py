from __future__ import annotations

import pytest

from OpenPinch.domain.fluids import (
    build_coolprop_abstract_state,
    critical_point,
    validate_coolprop_fluid_name,
)


def test_critical_point_of_a_fluid_spec_is_solved_once():
    """A mixture critical point takes about 0.1 s in CoolProp, and HPR
    targeting asks for it for every candidate cycle."""
    CoolProp = pytest.importorskip("CoolProp")
    spec = "R134a[0.5]&R245fa[0.5]"
    state = build_coolprop_abstract_state(spec)
    expected = (
        state.keyed_output(CoolProp.iT_critical),
        state.keyed_output(CoolProp.iP_critical),
        state.keyed_output(CoolProp.irhomass_critical),
    )
    critical_point.cache_clear()

    assert critical_point(spec) == pytest.approx(expected, rel=0.0, abs=0.0)
    assert critical_point(spec) == pytest.approx(expected, rel=0.0, abs=0.0)
    assert critical_point.cache_info().hits == 1


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
