"""Tests for prepared-zone service orchestration helpers."""

from __future__ import annotations

import pytest

from OpenPinch.classes.problem_table import ProblemTable
from OpenPinch.classes.zone import Zone
from OpenPinch.lib.config import Configuration
from OpenPinch.lib.enums import ProblemTableLabel as PT, TT, ZT
from OpenPinch.lib.target_schema import BaseTargetModel, DirectIntegrationTarget, TotalSiteTarget
from OpenPinch.services import services_entry as svc


def _make_zone() -> Zone:
    return Zone(name="Plant", type=ZT.S.value, zone_config=Configuration())


def _dummy_problem_table() -> ProblemTable:
    return ProblemTable({PT.T.value: [0.0]})


def _make_target(zone: Zone, target_type: str) -> BaseTargetModel:
    if target_type == TT.DI.value:
        return DirectIntegrationTarget(
            zone_name=zone.name,
            type=target_type,
            parent_zone=zone.parent_zone,
            config=zone.config,
            pt=_dummy_problem_table(),
            pt_real=_dummy_problem_table(),
            hot_utility_target=0.0,
            cold_utility_target=0.0,
            heat_recovery_target=0.0,
        )
    if target_type == TT.TS.value:
        return TotalSiteTarget(
            zone_name=zone.name,
            type=target_type,
            parent_zone=zone.parent_zone,
            config=zone.config,
            pt=_dummy_problem_table(),
            pt_real=_dummy_problem_table(),
            hot_utility_target=0.0,
            cold_utility_target=0.0,
            heat_recovery_target=0.0,
        )
    return BaseTargetModel(
        zone_name=zone.name,
        type=target_type,
        parent_zone=zone.parent_zone,
        config=zone.config,
    )


@pytest.mark.parametrize(
    ("service_name", "target_id", "expected_is_heat_pumping"),
    [
        (
            "direct_heat_pump_service",
            TT.DHP.value,
            True,
        ),
        (
            "direct_refrigeration_service",
            TT.DR.value,
            False,
        ),
    ],
)
def test_direct_hpr_services_enable_flags_and_bootstrap_direct_integration(
    monkeypatch,
    service_name: str,
    target_id: str,
    expected_is_heat_pumping: bool,
):
    zone = _make_zone()
    calls = {"direct": 0}

    def fake_direct_heat_integration_service(target_zone: Zone) -> Zone:
        calls["direct"] += 1
        target_zone.add_target(_make_target(target_zone, TT.DI.value))
        return target_zone

    def fake_compute(target_zone: Zone, is_heat_pumping: bool) -> BaseTargetModel:
        assert is_heat_pumping is expected_is_heat_pumping
        return _make_target(target_zone, target_id)

    monkeypatch.setattr(
        svc,
        "direct_heat_integration_service",
        fake_direct_heat_integration_service,
    )
    monkeypatch.setattr(
        svc,
        "compute_direct_heat_pump_or_refrigeration_target",
        fake_compute,
    )

    out = getattr(svc, service_name)(zone)

    assert out is zone
    assert calls["direct"] == 1
    assert TT.DI.value in zone.targets
    assert target_id in zone.targets


@pytest.mark.parametrize(
    ("service_name", "target_id", "expected_is_heat_pumping"),
    [
        (
            "indirect_heat_pump_service",
            TT.IHP.value,
            True,
        ),
        (
            "indirect_refrigeration_service",
            TT.IR.value,
            False,
        ),
    ],
)
def test_indirect_hpr_services_enable_flags_and_bootstrap_total_site_targets(
    monkeypatch,
    service_name: str,
    target_id: str,
    expected_is_heat_pumping: bool,
):
    zone = _make_zone()
    calls = {"indirect": 0}

    def fake_indirect_heat_integration_service(target_zone: Zone) -> Zone:
        calls["indirect"] += 1
        target_zone.add_target(_make_target(target_zone, TT.TS.value))
        return target_zone

    def fake_compute(target_zone: Zone, is_heat_pumping: bool) -> BaseTargetModel:
        assert is_heat_pumping is expected_is_heat_pumping
        return _make_target(target_zone, target_id)

    monkeypatch.setattr(
        svc,
        "indirect_heat_integration_service",
        fake_indirect_heat_integration_service,
    )
    monkeypatch.setattr(
        svc,
        "compute_indirect_heat_pump_or_refrigeration_target",
        fake_compute,
    )

    out = getattr(svc, service_name)(zone)

    assert out is zone
    assert calls["indirect"] == 1
    assert TT.TS.value in zone.targets
    assert target_id in zone.targets


def test_indirect_service_skips_none_targets(monkeypatch):
    zone = _make_zone()

    monkeypatch.setattr(
        svc,
        "indirect_heat_integration_service",
        lambda target_zone: (
            target_zone.add_target(_make_target(target_zone, TT.TS.value))
            or target_zone
        ),
    )
    monkeypatch.setattr(
        svc,
        "compute_indirect_heat_pump_or_refrigeration_target",
        lambda target_zone, is_heat_pumping: None,
    )

    out = svc.indirect_refrigeration_service(zone)

    assert out is zone
    assert TT.TS.value in zone.targets
    assert TT.IR.value not in zone.targets


def test_cogeneration_and_area_cost_services_update_direct_integration_target(
    monkeypatch,
):
    zone = _make_zone()
    calls = {"direct": 0, "cogen": 0}

    def fake_direct_heat_integration_service(target_zone: Zone) -> Zone:
        calls["direct"] += 1
        di_target = target_zone.targets.get(TT.DI.value) or _make_target(
            target_zone,
            TT.DI.value,
        )
        target_zone.add_target(di_target)
        return target_zone

    def fake_get_power_cogeneration_above_pinch(target):
        calls["cogen"] += 1
        target.work_target = 12.0
        target.turbine_efficiency_target = 0.42
        return target

    monkeypatch.setattr(
        svc,
        "direct_heat_integration_service",
        fake_direct_heat_integration_service,
    )
    monkeypatch.setattr(
        svc,
        "get_power_cogeneration_above_pinch",
        fake_get_power_cogeneration_above_pinch,
    )

    cogeneration_out = svc.power_cogeneration_service(zone)
    area_out = svc.area_cost_targeting_service(zone)

    assert cogeneration_out is zone
    assert area_out is zone
    assert calls["direct"] == 2
    assert calls["cogen"] == 1
    assert zone.targets[TT.DI.value].work_target == 12.0
    assert zone.targets[TT.DI.value].turbine_efficiency_target == 0.42
