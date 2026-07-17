"""Tests for default targeting selector dispatch planning."""

from __future__ import annotations

import pytest

from OpenPinch.application._problem.targeting import dispatch as td
from OpenPinch.application._problem.targeting.plan import (
    TARGETING_METHOD_SPECS,
    TargetingMethodSpec,
    TargetingPlan,
    build_targeting_plan,
)
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.configuration_fields import CONFIG_FIELD_SPECS
from OpenPinch.domain.enums import TT, ZT
from OpenPinch.domain.zone import Zone


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


def test_targeting_dispatch_rejects_unknown_zone_type_and_ignores_noncallable_service():
    zone = Zone(name="Unknown", type="Unknown")

    with pytest.raises(ValueError, match="No valid zone"):
        td.run_targeting_for_zone_and_subzones(zone)

    assert td._invoke_service(None, zone) is None


def test_targeting_dispatch_invokes_direct_service_for_nested_operation():
    calls = []
    config = Configuration(
        options=_targeting_false_options()
        | {"TARGETING_DIRECT_OPERATION_ENABLED": True}
    )
    parent = Zone(name="ParentOperation", type=ZT.O.value, config=config)
    child = Zone(name="ChildOperation", type=ZT.O.value, config=config)
    parent.add_zone(child)

    td.run_targeting_for_zone_and_subzones(
        parent,
        direct_service_func=lambda zone, args=None: calls.append(zone.name),
    )

    assert calls == ["ChildOperation", "ParentOperation"]


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


def test_targeting_plan_deduplicates_repeated_services():
    calls = []

    def direct_service(zone, args=None):
        calls.append(zone.name)
        return zone

    specs = (
        TargetingMethodSpec(
            selector="FIRST",
            service=direct_service,
            target_type=TT.DI.value,
            slot="direct",
            prerequisites=(),
            zone_applicability=(ZT.S.value,),
            execution_order=10,
        ),
        TargetingMethodSpec(
            selector="SECOND",
            service=direct_service,
            target_type=TT.DI.value,
            slot="direct",
            prerequisites=(),
            zone_applicability=(ZT.S.value,),
            execution_order=20,
        ),
    )
    plan = TargetingPlan(specs=specs)
    site = Zone(name="Site", type=ZT.S.value)

    assert plan.direct_services == (direct_service,)
    plan.composite_direct_service()(site)

    assert calls == ["Site"]
