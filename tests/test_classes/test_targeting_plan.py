"""Tests for default targeting selector dispatch planning."""

from __future__ import annotations

from OpenPinch.classes._problem import _target_dispatch as td
from OpenPinch.classes._problem._target_plan import (
    TARGETING_METHOD_SPECS,
    TargetingMethodSpec,
    TargetingPlan,
    build_targeting_plan,
)
from OpenPinch.classes.zone import Zone
from OpenPinch.lib.config import Configuration
from OpenPinch.lib.config_metadata import CONFIG_FIELD_SPECS
from OpenPinch.lib.enums import TT, ZT


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


def test_targeting_plan_direct_operation_composite_skips_site_and_process_zones():
    calls = []

    def direct_service(zone, args=None):
        calls.append(zone.name)
        return zone

    plan = TargetingPlan(
        specs=(
            TargetingMethodSpec(
                selector="TARGETING_DIRECT_OPERATION_ENABLED",
                service=direct_service,
                target_type=TT.DI.value,
                slot="direct",
                prerequisites=(),
                zone_applicability=(ZT.O.value,),
                execution_order=20,
            ),
        )
    )
    config = Configuration(
        options=_targeting_false_options()
        | {"TARGETING_DIRECT_OPERATION_ENABLED": True}
    )
    site = Zone(name="Site", type=ZT.S.value, config=config)
    process = Zone(name="Process", type=ZT.P.value, config=config)
    operation = Zone(name="Operation", type=ZT.O.value, config=config)
    process.add_zone(operation)
    site.add_zone(process)

    td.run_targeting_for_zone_and_subzones(
        site,
        direct_service_func=plan.composite_direct_service(),
    )

    assert calls == ["Operation"]


def test_targeting_plan_utility_composite_runs_on_process_without_ts_selector():
    calls = []

    def utility_service(zone, args=None):
        calls.append(zone.name)
        return zone

    plan = TargetingPlan(
        specs=(
            TargetingMethodSpec(
                selector="TARGETING_UTILITY_HP_ENABLED",
                service=utility_service,
                target_type=TT.IHP.value,
                slot="indirect",
                prerequisites=(TT.TS.value,),
                zone_applicability=(ZT.S.value, ZT.P.value),
                execution_order=60,
            ),
        )
    )
    config = Configuration(
        options=_targeting_false_options() | {"TARGETING_UTILITY_HP_ENABLED": True}
    )
    process = Zone(name="Process", type=ZT.P.value, config=config)
    operation = Zone(name="Operation", type=ZT.O.value, config=config)
    process.add_zone(operation)

    td.run_targeting_for_zone_and_subzones(
        process,
        indirect_service_func=plan.composite_indirect_service(),
    )

    assert calls == ["Process"]
