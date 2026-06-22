"""Tests for default targeting selector dispatch planning."""

from __future__ import annotations

from OpenPinch.classes._problem._target_plan import (
    TARGETING_METHOD_SPECS,
    build_targeting_plan,
)
from OpenPinch.lib.config import Configuration
from OpenPinch.lib.config_metadata import CONFIG_FIELD_SPECS


def _targeting_false_options() -> dict[str, bool]:
    return {
        key: False
        for key in CONFIG_FIELD_SPECS
        if key.startswith("TARGETING_") and key.endswith("_ENABLED")
    }


def test_targeting_plan_has_spec_for_every_targeting_selector():
    selectors = {
        key
        for key in CONFIG_FIELD_SPECS
        if key.startswith("TARGETING_") and key.endswith("_ENABLED")
    }

    assert {spec.selector for spec in TARGETING_METHOD_SPECS} == selectors


def test_targeting_plan_dispatches_each_selector_to_one_service():
    base_options = _targeting_false_options()

    for expected in TARGETING_METHOD_SPECS:
        cfg = Configuration(options=base_options | {expected.selector: True})
        plan = build_targeting_plan(cfg)

        assert plan.specs == (expected,)
        assert callable(expected.service)
        if expected.slot == "direct":
            assert plan.direct_services == (expected.service,)
            assert plan.indirect_services == ()
        else:
            assert plan.indirect_services == (expected.service,)
            assert plan.direct_services == ()


def test_targeting_plan_default_runs_direct_site_targeting():
    plan = build_targeting_plan(Configuration())

    assert [spec.selector for spec in plan.specs] == ["TARGETING_DIRECT_SITE_ENABLED"]
    assert plan.composite_direct_service() is not None
    assert plan.composite_indirect_service() is None
