"""Tests for prepared-zone service orchestration helpers."""

from __future__ import annotations

import pytest

import OpenPinch.application.targeting as svc
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.enums import TT, ZT
from OpenPinch.domain.enums import ProblemTableLabel as PT
from OpenPinch.domain.problem_table import ProblemTable
from OpenPinch.domain.stream_collection import StreamCollection
from OpenPinch.domain.targets import (
    BaseTargetModel,
    DirectHeatPumpTarget,
    DirectIntegrationTarget,
    DirectRefrigerationTarget,
    EnergyTransferTarget,
    IndirectHeatPumpTarget,
    IndirectRefrigerationTarget,
    TotalSiteTarget,
)
from OpenPinch.domain.zone import Zone


def _make_zone() -> Zone:
    return Zone(name="Plant", type=ZT.S.value, config=Configuration())


def _dummy_problem_table() -> ProblemTable:
    return ProblemTable({PT.T: [0.0]})


def _make_target(
    zone: Zone,
    target_type: str,
    *,
    period_id: str | None = None,
    period_idx: int | None = None,
) -> BaseTargetModel:
    if target_type == TT.DI.value:
        return DirectIntegrationTarget(
            zone_name=zone.name,
            period_id=period_id,
            period_idx=period_idx,
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
            period_id=period_id,
            period_idx=period_idx,
            type=target_type,
            parent_zone=zone.parent_zone,
            config=zone.config,
            pt=_dummy_problem_table(),
            hot_utility_target=0.0,
            cold_utility_target=0.0,
            heat_recovery_target=0.0,
        )
    if target_type == TT.ET.value:
        return EnergyTransferTarget(
            zone_name=zone.name,
            period_id=period_id,
            period_idx=period_idx,
            type=target_type,
            parent_zone=zone.parent_zone,
            config=zone.config,
            pt=_dummy_problem_table(),
            hot_utility_target=0.0,
            cold_utility_target=0.0,
            heat_recovery_target=0.0,
            base_target_type=TT.DI.value,
            base_target_name=f"{zone.name}/{TT.DI.value}",
        )
    if target_type in {
        TT.DHP.value,
        TT.IHP.value,
        TT.DR.value,
        TT.IR.value,
    }:
        model_cls = {
            TT.DHP.value: DirectHeatPumpTarget,
            TT.IHP.value: IndirectHeatPumpTarget,
            TT.DR.value: DirectRefrigerationTarget,
            TT.IR.value: IndirectRefrigerationTarget,
        }[target_type]
        return model_cls(
            zone_name=zone.name,
            period_id=period_id,
            period_idx=period_idx,
            type=target_type,
            parent_zone=zone.parent_zone,
            config=zone.config,
            pt=_dummy_problem_table(),
            hot_utilities=StreamCollection(),
            cold_utilities=StreamCollection(),
            hot_utility_target=0.0,
            cold_utility_target=0.0,
            heat_recovery_target=0.0,
            hpr_cycle="stub",
            hpr_utility_total=0.0,
            hpr_work=0.0,
            hpr_external_utility=0.0,
            hpr_ambient_hot=0.0,
            hpr_ambient_cold=0.0,
            hpr_cop=0.0,
            hpr_eta_he=0.0,
            hpr_success=True,
            hpr_hot_streams=StreamCollection(),
            hpr_cold_streams=StreamCollection(),
            hpr_details={},
        )
    return BaseTargetModel(
        zone_name=zone.name,
        period_id=period_id,
        period_idx=period_idx,
        type=target_type,
        parent_zone=zone.parent_zone,
        config=zone.config,
    )


def test_data_preprocessing_service_delegates_target_input_payload(monkeypatch):
    expected_zone = _make_zone()
    captured = {}
    input_data = type(
        "StaticTargetInput",
        (),
        {
            "streams": [{"name": "H1"}],
            "utilities": [{"name": "Steam"}],
            "options": {"THERMAL_DT_CONT": 10.0},
            "zone_tree": {"Site": ["Area"]},
        },
    )()

    def fake_prepare_problem(**kwargs):
        captured.update(kwargs)
        return expected_zone

    monkeypatch.setattr(svc, "prepare_problem", fake_prepare_problem)

    zone = svc.data_preprocessing_service(input_data, project_name="Static Site")

    assert zone is expected_zone
    assert captured == {
        "project_name": "Static Site",
        "streams": input_data.streams,
        "utilities": input_data.utilities,
        "options": input_data.options,
        "zone_tree": input_data.zone_tree,
    }


def test_indirect_heat_integration_service_records_total_process_and_site_targets(
    monkeypatch,
):
    zone = _make_zone()
    calls = []

    def fake_compute_total(target_zone: Zone, args: dict | None = None):
        calls.append(("total_process", args))
        return _make_target(target_zone, TT.TZ.value)

    def fake_compute_indirect(target_zone: Zone, args: dict | None = None):
        calls.append(("total_site", args))
        return _make_target(target_zone, TT.TS.value)

    monkeypatch.setattr(
        svc, "compute_total_subzone_utility_targets", fake_compute_total
    )
    monkeypatch.setattr(
        svc, "compute_indirect_integration_targets", fake_compute_indirect
    )
    monkeypatch.setattr(
        svc,
        "apply_exergy_if_enabled",
        lambda target, target_zone, apply_func: target,
    )

    out = svc.indirect_heat_integration_service(zone, args={"period_idx": 0})

    assert out is zone
    assert calls == [
        ("total_process", {"period_idx": 0}),
        ("total_site", {"period_idx": 0}),
    ]
    assert TT.TZ.value in zone.targets
    assert TT.TS.value in zone.targets


@pytest.mark.parametrize(
    ("service_name", "stale_target_type", "base_target_type", "compute_attr"),
    [
        (
            "direct_heat_pump_service",
            TT.DHP.value,
            TT.DI.value,
            "compute_direct_heat_pump_or_refrigeration_target",
        ),
        (
            "direct_refrigeration_service",
            TT.DR.value,
            TT.DI.value,
            "compute_direct_heat_pump_or_refrigeration_target",
        ),
        (
            "indirect_heat_pump_service",
            TT.IHP.value,
            TT.TS.value,
            "compute_indirect_heat_pump_or_refrigeration_target",
        ),
    ],
)
def test_hpr_services_remove_stale_target_when_solver_returns_none(
    monkeypatch,
    service_name: str,
    stale_target_type: str,
    base_target_type: str,
    compute_attr: str,
):
    zone = _make_zone()
    zone.add_target(_make_target(zone, base_target_type))
    zone.add_target(_make_target(zone, stale_target_type))

    monkeypatch.setattr(
        svc,
        compute_attr,
        lambda target_zone, is_heat_pumping, args=None: None,
    )

    out = getattr(svc, service_name)(zone)

    assert out is zone
    assert base_target_type in zone.targets
    assert stale_target_type not in zone.targets


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

    def fake_direct_heat_integration_service(
        target_zone: Zone,
        args: dict | None = None,
    ) -> Zone:
        calls["direct"] += 1
        assert args is None
        target_zone.add_target(_make_target(target_zone, TT.DI.value))
        return target_zone

    def fake_compute(
        target_zone: Zone,
        is_heat_pumping: bool,
        args: dict | None = None,
    ) -> BaseTargetModel:
        assert is_heat_pumping is expected_is_heat_pumping
        assert args is None
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

    def fake_indirect_heat_integration_service(
        target_zone: Zone,
        args: dict | None = None,
    ) -> Zone:
        calls["indirect"] += 1
        assert args is None
        target_zone.add_target(_make_target(target_zone, TT.TS.value))
        return target_zone

    def fake_compute(
        target_zone: Zone,
        is_heat_pumping: bool,
        args: dict | None = None,
    ) -> BaseTargetModel:
        assert is_heat_pumping is expected_is_heat_pumping
        assert args is None
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


def test_exergy_targeting_service_auto_selects_total_site_first(monkeypatch):
    zone = _make_zone()
    zone.add_target(_make_target(zone, TT.TS.value))
    zone.add_target(_make_target(zone, TT.DI.value))
    enriched = []

    monkeypatch.setattr(
        svc,
        "apply_exergy_targeting",
        lambda target: enriched.append(target.type) or target,
    )

    out = svc.exergy_targeting_service(zone)

    assert out is zone
    assert enriched == [TT.TS.value]
    assert zone._selected_exergy_target_type == TT.TS.value


def test_exergy_targeting_service_honours_explicit_base_target_type(monkeypatch):
    zone = _make_zone()
    zone.add_target(_make_target(zone, TT.TS.value))
    zone.add_target(_make_target(zone, TT.DI.value))
    enriched = []

    monkeypatch.setattr(
        svc,
        "apply_exergy_targeting",
        lambda target: enriched.append(target.type) or target,
    )

    out = svc.exergy_targeting_service(
        zone,
        args={"base_target_type": TT.DI.value},
    )

    assert out is zone
    assert enriched == [TT.DI.value]
    assert zone._selected_exergy_target_type == TT.DI.value


def test_exergy_targeting_service_uses_existing_matching_state_only(monkeypatch):
    zone = _make_zone()
    zone.set_period_context({"0": 0, "peak": 1}, [1.0, 1.0], 2)
    zone.add_target(_make_target(zone, TT.TS.value, period_id="0", period_idx=0))
    zone.add_target(_make_target(zone, TT.DI.value, period_id="peak", period_idx=1))
    enriched = []

    monkeypatch.setattr(
        svc,
        "apply_exergy_targeting",
        lambda target: enriched.append((target.type, target.period_id)) or target,
    )

    out = svc.exergy_targeting_service(zone, {"period_id": "peak"})

    assert out is zone
    assert enriched == [(TT.DI.value, "peak")]
    assert zone._selected_exergy_target_type == TT.DI.value


def test_exergy_targeting_service_explicit_target_raises_when_unavailable():
    zone = _make_zone()
    zone.add_target(_make_target(zone, TT.DI.value))

    with pytest.raises(RuntimeError, match="requires an existing target"):
        svc.exergy_targeting_service(zone, {"base_target_type": TT.TS.value})


def test_exergy_targeting_service_raises_when_no_existing_target():
    zone = _make_zone()

    with pytest.raises(RuntimeError, match="compatible existing target"):
        svc.exergy_targeting_service(zone)


def test_direct_heat_integration_service_skips_exergy_when_disabled(monkeypatch):
    zone = _make_zone()
    called = {"exergy": 0}

    monkeypatch.setattr(
        svc,
        "compute_direct_integration_targets",
        lambda target_zone, args=None: _make_target(target_zone, TT.DI.value),
    )
    monkeypatch.setattr(
        svc,
        "apply_exergy_targeting",
        lambda target: called.__setitem__("exergy", called["exergy"] + 1) or target,
    )

    out = svc.direct_heat_integration_service(zone)

    assert out is zone
    assert called["exergy"] == 0


def test_exergy_targeting_service_skips_site_only_targets_on_leaf_zone(monkeypatch):
    zone = _make_zone()
    zone.add_target(_make_target(zone, TT.DI.value))
    enriched = []

    monkeypatch.setattr(
        svc,
        "apply_exergy_targeting",
        lambda target: enriched.append(target.type) or target,
    )

    out = svc.exergy_targeting_service(zone)

    assert out is zone
    assert enriched == [TT.DI.value]
    assert zone._selected_exergy_target_type == TT.DI.value


def test_exergy_targeting_service_does_not_refresh_state_mismatched_target(
    monkeypatch,
):
    zone = _make_zone()
    zone.set_period_context({"0": 0, "peak": 1}, [1.0, 1.0], 2)
    zone.add_target(_make_target(zone, TT.TS.value, period_id="0", period_idx=0))

    monkeypatch.setattr(
        svc,
        "apply_exergy_targeting",
        lambda target: pytest.fail("period-mismatched target should not be enriched"),
    )

    with pytest.raises(RuntimeError, match="requires an existing target"):
        svc.exergy_targeting_service(
            zone,
            {"base_target_type": TT.TS.value, "period_id": "peak"},
        )


def test_indirect_service_skips_none_targets(monkeypatch):
    zone = _make_zone()

    monkeypatch.setattr(
        svc,
        "indirect_heat_integration_service",
        lambda target_zone, args=None: (
            target_zone.add_target(_make_target(target_zone, TT.TS.value))
            or target_zone
        ),
    )
    monkeypatch.setattr(
        svc,
        "compute_indirect_heat_pump_or_refrigeration_target",
        lambda target_zone, is_heat_pumping, args=None: None,
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

    def fake_direct_heat_integration_service(
        target_zone: Zone,
        args: dict | None = None,
    ) -> Zone:
        calls["direct"] += 1
        assert args in (
            None,
            {
                "base_target_type": TT.DI.value,
                "period_idx": 0,
            },
        )
        di_target = target_zone.targets.get(TT.DI.value) or _make_target(
            target_zone,
            TT.DI.value,
        )
        target_zone.add_target(di_target)
        return target_zone

    def fake_get_power_cogeneration_above_pinch(target, args=None):
        calls["cogen"] += 1
        assert args == {
            "base_target_type": TT.DI.value,
            "period_idx": 0,
        }
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

    cogeneration_out = svc.power_cogeneration_service(
        zone,
        {"base_target_type": TT.DI.value},
    )
    area_out = svc.area_cost_targeting_service(zone)

    assert cogeneration_out is zone
    assert area_out is zone
    assert calls["direct"] == 2
    assert calls["cogen"] == 1
    assert zone.targets[TT.DI.value].work_target == 12.0
    assert zone.targets[TT.DI.value].turbine_efficiency_target == 0.42
    assert zone._selected_cogeneration_target_type == TT.DI.value


def test_energy_transfer_service_bootstraps_direct_target_on_leaf(monkeypatch):
    zone = _make_zone()
    calls = {"direct": 0}

    def fake_direct_heat_integration_service(
        target_zone: Zone,
        args: dict | None = None,
    ) -> Zone:
        calls["direct"] += 1
        target_zone.add_target(_make_target(target_zone, TT.DI.value))
        return target_zone

    monkeypatch.setattr(
        svc,
        "direct_heat_integration_service",
        fake_direct_heat_integration_service,
    )

    out = svc.energy_transfer_analysis_service(zone)

    assert out is zone
    assert calls["direct"] == 1
    assert TT.ET.value in zone.targets
    assert zone.targets[TT.ET.value].base_target_type == TT.DI.value
    assert zone._selected_energy_transfer_base_target_type == TT.DI.value


def test_energy_transfer_service_prefers_existing_total_site_target():
    zone = _make_zone()
    zone.add_target(_make_target(zone, TT.TS.value))
    zone.add_target(_make_target(zone, TT.DI.value))

    out = svc.energy_transfer_analysis_service(zone)

    assert out is zone
    assert zone.targets[TT.ET.value].base_target_type == TT.TS.value
    assert zone._selected_energy_transfer_base_target_type == TT.TS.value


def test_energy_transfer_total_site_bootstrap_solves_parent_and_child_di(
    monkeypatch,
):
    zone = _make_zone()
    subzone = Zone(name="Bleaching", type=ZT.P.value, config=zone.config)
    zone.add_zone(subzone)
    calls: list[tuple[str, str]] = []

    def fake_direct_heat_integration_service(
        target_zone: Zone,
        args: dict | None = None,
    ) -> Zone:
        calls.append((TT.DI.value, target_zone.name))
        target_zone.add_target(_make_target(target_zone, TT.DI.value))
        return target_zone

    def fake_indirect_heat_integration_service(
        target_zone: Zone,
        args: dict | None = None,
    ) -> Zone:
        calls.append((TT.TS.value, target_zone.name))
        assert TT.DI.value in target_zone.targets
        assert TT.DI.value in subzone.targets
        target_zone.add_target(_make_target(target_zone, TT.TS.value))
        return target_zone

    monkeypatch.setattr(
        svc,
        "direct_heat_integration_service",
        fake_direct_heat_integration_service,
    )
    monkeypatch.setattr(
        svc,
        "indirect_heat_integration_service",
        fake_indirect_heat_integration_service,
    )

    out = svc.energy_transfer_analysis_service(
        zone,
        {"base_target_type": TT.TS.value},
    )

    assert out is zone
    assert calls == [
        (TT.DI.value, "Plant"),
        (TT.DI.value, "Bleaching"),
        (TT.TS.value, "Plant"),
    ]
    assert zone.targets[TT.ET.value].base_target_type == TT.TS.value


def test_energy_transfer_total_site_uses_one_subzone_layer_of_gccs(monkeypatch):
    zone = _make_zone()
    area = Zone(name="Area", type=ZT.S.value, config=zone.config)
    unit = Zone(name="Unit", type=ZT.P.value, config=zone.config)
    zone.add_zone(area)
    area.add_zone(unit)
    calls: list[tuple[str, str]] = []
    source_names: list[str] = []

    def fake_direct_heat_integration_service(
        target_zone: Zone,
        args: dict | None = None,
    ) -> Zone:
        calls.append((TT.DI.value, target_zone.name))
        target_zone.add_target(_make_target(target_zone, TT.DI.value))
        return target_zone

    def fake_indirect_heat_integration_service(
        target_zone: Zone,
        args: dict | None = None,
    ) -> Zone:
        calls.append((TT.TS.value, target_zone.name))
        assert TT.DI.value in target_zone.targets
        assert TT.DI.value in area.targets
        assert TT.DI.value not in unit.targets
        target_zone.add_target(_make_target(target_zone, TT.TS.value))
        return target_zone

    def fake_compute_energy_transfer_target(base_target, source_targets=None):
        for source in source_targets:
            source_names.append(source["name"])
            assert source["target"].zone_name == "Area"
        target = _make_target(zone, TT.ET.value)
        target.base_target_type = base_target.type
        target.base_target_name = base_target.name
        return target

    monkeypatch.setattr(
        svc,
        "direct_heat_integration_service",
        fake_direct_heat_integration_service,
    )
    monkeypatch.setattr(
        svc,
        "indirect_heat_integration_service",
        fake_indirect_heat_integration_service,
    )
    monkeypatch.setattr(
        svc,
        "compute_energy_transfer_target",
        fake_compute_energy_transfer_target,
    )

    out = svc.energy_transfer_analysis_service(
        zone,
        {"base_target_type": TT.TS.value},
    )

    assert out is zone
    assert calls == [
        (TT.DI.value, "Plant"),
        (TT.DI.value, "Area"),
        (TT.TS.value, "Plant"),
    ]
    assert source_names == ["Plant/Area"]
    assert zone.targets[TT.ET.value].base_target_type == TT.TS.value


def test_cogeneration_service_prefers_total_site_before_direct_integration(
    monkeypatch,
):
    zone = _make_zone()
    zone.add_target(_make_target(zone, TT.TS.value))
    zone.add_target(_make_target(zone, TT.DI.value))
    selected_targets: list[str] = []

    def fake_get_power_cogeneration_above_pinch(target, args=None):
        selected_targets.append(target.type)
        target.work_target = 8.0
        return target

    monkeypatch.setattr(
        svc,
        "get_power_cogeneration_above_pinch",
        fake_get_power_cogeneration_above_pinch,
    )

    out = svc.power_cogeneration_service(zone)

    assert out is zone
    assert selected_targets == [TT.TS.value]
    assert zone.targets[TT.TS.value].work_target == 8.0
    assert zone.targets[TT.DI.value].work_target is None
    assert zone._selected_cogeneration_target_type == TT.TS.value


def test_cogeneration_service_falls_back_from_total_site_to_indirect_heat_pump(
    monkeypatch,
):
    zone = _make_zone()
    refresh_order: list[str] = []
    selected_targets: list[str] = []

    def fake_indirect_heat_integration_service(target_zone: Zone, args=None) -> Zone:
        refresh_order.append(TT.TS.value)
        return target_zone

    def fake_indirect_heat_pump_service(target_zone: Zone, args=None) -> Zone:
        refresh_order.append(TT.IHP.value)
        target_zone.add_target(_make_target(target_zone, TT.IHP.value))
        return target_zone

    def fake_get_power_cogeneration_above_pinch(target, args=None):
        selected_targets.append(target.type)
        return target

    monkeypatch.setattr(
        svc,
        "indirect_heat_integration_service",
        fake_indirect_heat_integration_service,
    )
    monkeypatch.setattr(
        svc,
        "indirect_heat_pump_service",
        fake_indirect_heat_pump_service,
    )
    monkeypatch.setattr(
        svc,
        "get_power_cogeneration_above_pinch",
        fake_get_power_cogeneration_above_pinch,
    )

    out = svc.power_cogeneration_service(zone)

    assert out is zone
    assert refresh_order == [TT.TS.value, TT.IHP.value]
    assert selected_targets == [TT.IHP.value]
    assert zone._selected_cogeneration_target_type == TT.IHP.value


def test_cogeneration_service_falls_back_to_direct_integration_when_needed(
    monkeypatch,
):
    zone = _make_zone()
    refresh_order: list[str] = []
    selected_targets: list[str] = []

    def _missing(target_name: str):
        def _service(target_zone: Zone, args=None) -> Zone:
            refresh_order.append(target_name)
            return target_zone

        return _service

    def fake_direct_heat_integration_service(target_zone: Zone, args=None) -> Zone:
        refresh_order.append(TT.DI.value)
        target_zone.add_target(_make_target(target_zone, TT.DI.value))
        return target_zone

    def fake_get_power_cogeneration_above_pinch(target, args=None):
        selected_targets.append(target.type)
        return target

    monkeypatch.setattr(
        svc,
        "indirect_heat_integration_service",
        _missing(TT.TS.value),
    )
    monkeypatch.setattr(
        svc,
        "indirect_heat_pump_service",
        _missing(TT.IHP.value),
    )
    monkeypatch.setattr(
        svc,
        "indirect_refrigeration_service",
        _missing(TT.IR.value),
    )
    monkeypatch.setattr(
        svc,
        "direct_heat_pump_service",
        _missing(TT.DHP.value),
    )
    monkeypatch.setattr(
        svc,
        "direct_refrigeration_service",
        _missing(TT.DR.value),
    )
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

    out = svc.power_cogeneration_service(zone)

    assert out is zone
    assert refresh_order == [
        TT.TS.value,
        TT.IHP.value,
        TT.IR.value,
        TT.DHP.value,
        TT.DR.value,
        TT.DI.value,
    ]
    assert selected_targets == [TT.DI.value]
    assert zone._selected_cogeneration_target_type == TT.DI.value


@pytest.mark.parametrize(
    "base_target_type",
    [TT.TS.value, TT.IHP.value, TT.DI.value],
)
def test_cogeneration_service_explicit_override_targets_exact_family(
    monkeypatch,
    base_target_type: str,
):
    zone = _make_zone()
    for target_type in (
        TT.TS.value,
        TT.IHP.value,
        TT.IR.value,
        TT.DHP.value,
        TT.DR.value,
        TT.DI.value,
    ):
        zone.add_target(_make_target(zone, target_type))

    selected_targets: list[str] = []

    def fake_get_power_cogeneration_above_pinch(target, args=None):
        selected_targets.append(target.type)
        return target

    monkeypatch.setattr(
        svc,
        "get_power_cogeneration_above_pinch",
        fake_get_power_cogeneration_above_pinch,
    )

    out = svc.power_cogeneration_service(
        zone,
        {"base_target_type": base_target_type},
    )

    assert out is zone
    assert selected_targets == [base_target_type]
    assert zone._selected_cogeneration_target_type == base_target_type


def test_cogeneration_service_rejects_unsupported_explicit_override():
    zone = _make_zone()

    with pytest.raises(ValueError, match="Unsupported cogeneration base_target_type"):
        svc.power_cogeneration_service(zone, {"base_target_type": TT.TZ.value})


def test_cogeneration_service_explicit_target_raises_when_unavailable(monkeypatch):
    zone = _make_zone()

    monkeypatch.setattr(
        svc,
        "indirect_heat_integration_service",
        lambda target_zone, args=None: target_zone,
    )

    with pytest.raises(
        RuntimeError,
        match="could not produce target 'Total Site Target'",
    ):
        svc.power_cogeneration_service(zone, {"base_target_type": TT.TS.value})


def test_cogeneration_service_refreshes_state_mismatched_target(monkeypatch):
    zone = _make_zone()
    zone.set_period_context({"peak": 0, "shoulder": 1}, weights=None, num_periods=2)
    zone.add_target(
        _make_target(
            zone,
            TT.TS.value,
            period_id="shoulder",
            period_idx=1,
        )
    )
    selected_periods: list[str | None] = []
    refresh_calls = {"count": 0}

    def fake_indirect_heat_integration_service(target_zone: Zone, args=None) -> Zone:
        refresh_calls["count"] += 1
        target_zone.add_target(
            _make_target(
                target_zone,
                TT.TS.value,
                period_id="peak",
                period_idx=0,
            )
        )
        return target_zone

    def fake_get_power_cogeneration_above_pinch(target, args=None):
        selected_periods.append(target.period_id)
        return target

    monkeypatch.setattr(
        svc,
        "indirect_heat_integration_service",
        fake_indirect_heat_integration_service,
    )
    monkeypatch.setattr(
        svc,
        "get_power_cogeneration_above_pinch",
        fake_get_power_cogeneration_above_pinch,
    )

    out = svc.power_cogeneration_service(zone, {"period_id": "peak"})

    assert out is zone
    assert refresh_calls["count"] == 1
    assert selected_periods == ["peak"]
    assert zone._selected_cogeneration_target_type == TT.TS.value


def test_cogeneration_service_no_viable_stage_does_not_fall_back(monkeypatch):
    zone = _make_zone()
    zone.add_target(_make_target(zone, TT.TS.value))
    zone.add_target(_make_target(zone, TT.DI.value))
    selected_targets: list[str] = []

    def fake_get_power_cogeneration_above_pinch(target, args=None):
        selected_targets.append(target.type)
        return target

    monkeypatch.setattr(
        svc,
        "get_power_cogeneration_above_pinch",
        fake_get_power_cogeneration_above_pinch,
    )

    out = svc.power_cogeneration_service(zone)

    assert out is zone
    assert selected_targets == [TT.TS.value]
    assert zone.targets[TT.TS.value].work_target is None
    assert zone.targets[TT.DI.value].work_target is None
    assert zone._selected_cogeneration_target_type == TT.TS.value


def test_direct_service_records_selected_period_id_on_zone(monkeypatch):
    zone = _make_zone()
    zone.set_period_context({"0": 0, "peak": 1}, [1.0, 1.0], 2)
    monkeypatch.setattr(
        svc,
        "compute_direct_integration_targets",
        lambda target_zone, args=None: _make_target(target_zone, TT.DI.value),
    )

    out = svc.direct_heat_integration_service(zone, {"period_id": "peak"})

    assert out is zone
    assert zone._selected_period_id == "peak"
    assert zone._selected_period_idx == 1


def test_direct_heat_pump_service_refreshes_direct_integration_for_new_period(
    monkeypatch,
):
    zone = _make_zone()
    zone.set_period_context({"0": 0, "peak": 1}, [1.0, 1.0], 2)
    zone.targets[TT.DI.value] = _make_target(
        zone,
        TT.DI.value,
        period_id="0",
        period_idx=0,
    )
    calls = {"direct": 0}

    def fake_direct_heat_integration_service(
        target_zone: Zone,
        args: dict | None = None,
    ) -> Zone:
        calls["direct"] += 1
        target_zone.add_target(
            _make_target(target_zone, TT.DI.value, period_id="peak", period_idx=1)
        )
        return target_zone

    monkeypatch.setattr(
        svc,
        "direct_heat_integration_service",
        fake_direct_heat_integration_service,
    )
    monkeypatch.setattr(
        svc,
        "compute_direct_heat_pump_or_refrigeration_target",
        lambda target_zone, is_heat_pumping, args=None: _make_target(
            target_zone,
            TT.DHP.value,
            period_id="peak",
            period_idx=1,
        ),
    )

    out = svc.direct_heat_pump_service(zone, {"period_id": "peak", "period_idx": 1})

    assert out is zone
    assert calls["direct"] == 1
    assert zone.targets[TT.DI.value].period_idx == 1
    assert zone.targets[TT.DHP.value].period_idx == 1


def test_indirect_heat_pump_service_refreshes_total_site_for_new_period(monkeypatch):
    zone = _make_zone()
    zone.set_period_context({"0": 0, "peak": 1}, [1.0, 1.0], 2)
    zone.targets[TT.TS.value] = _make_target(
        zone,
        TT.TS.value,
        period_id="0",
        period_idx=0,
    )
    calls = {"indirect": 0}

    def fake_indirect_heat_integration_service(
        target_zone: Zone,
        args: dict | None = None,
    ) -> Zone:
        calls["indirect"] += 1
        target_zone.add_target(
            _make_target(target_zone, TT.TS.value, period_id="peak", period_idx=1)
        )
        return target_zone

    monkeypatch.setattr(
        svc,
        "indirect_heat_integration_service",
        fake_indirect_heat_integration_service,
    )
    monkeypatch.setattr(
        svc,
        "compute_indirect_heat_pump_or_refrigeration_target",
        lambda target_zone, is_heat_pumping, args=None: _make_target(
            target_zone,
            TT.IHP.value,
            period_id="peak",
            period_idx=1,
        ),
    )

    out = svc.indirect_heat_pump_service(zone, {"period_id": "peak", "period_idx": 1})

    assert out is zone
    assert calls["indirect"] == 1
    assert zone.targets[TT.TS.value].period_idx == 1
    assert zone.targets[TT.IHP.value].period_idx == 1
