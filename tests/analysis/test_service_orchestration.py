"""Tests for prepared-zone service orchestration helpers."""

from __future__ import annotations

import pytest

import OpenPinch.application.targeting as svc
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.enums import ProblemTableLabel as ProblemTableLabel
from OpenPinch.domain.enums import TargetType, ZoneType
from OpenPinch.domain.problem_table import ProblemTable
from OpenPinch.domain.stream_collection import StreamCollection
from OpenPinch.domain.targets import (
    BaseTargetModel,
    DirectHeatPumpTarget,
    DirectIntegrationTarget,
    DirectRefrigerationTarget,
    EnergyTransferTarget,
    IndirectHeatPumpTarget,
    IndirectIntegrationTarget,
    IndirectRefrigerationTarget,
)
from OpenPinch.domain.zone import Zone


def _make_zone() -> Zone:
    return Zone(name="Plant", type=ZoneType.S.value, config=Configuration())


def _dummy_problem_table() -> ProblemTable:
    return ProblemTable({ProblemTableLabel.T: [0.0]})


def _make_target(
    zone: Zone,
    target_type: str,
    *,
    period_id: str | None = None,
    period_idx: int | None = None,
) -> BaseTargetModel:
    if target_type == TargetType.DI.value:
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
    if target_type == TargetType.II.value:
        return IndirectIntegrationTarget(
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
    if target_type == TargetType.ET.value:
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
            base_target_type=TargetType.DI.value,
            base_target_name=f"{zone.name}/{TargetType.DI.value}",
        )
    if target_type in {
        TargetType.DHP.value,
        TargetType.IHP.value,
        TargetType.DR.value,
        TargetType.IR.value,
    }:
        model_cls = {
            TargetType.DHP.value: DirectHeatPumpTarget,
            TargetType.IHP.value: IndirectHeatPumpTarget,
            TargetType.DR.value: DirectRefrigerationTarget,
            TargetType.IR.value: IndirectRefrigerationTarget,
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
    zone.add_target(_make_target(zone, TargetType.DI.value, period_idx=0))
    calls = []

    def fake_compute_total(target_zone: Zone, args: dict | None = None):
        calls.append(("total_process", args))
        return _make_target(target_zone, TargetType.SA.value)

    def fake_compute_indirect(target_zone: Zone, args: dict | None = None):
        calls.append(("total_site", args))
        return _make_target(target_zone, TargetType.II.value)

    monkeypatch.setattr(svc, "compute_subzone_aggregate_target", fake_compute_total)
    monkeypatch.setattr(
        svc, "compute_indirect_integration_targets", fake_compute_indirect
    )
    monkeypatch.setattr(
        zone,
        "import_hot_and_cold_streams_from_sub_zones",
        lambda **_kwargs: pytest.fail(
            "indirect targeting must not overwrite the zone's direct profiles"
        ),
    )
    out = svc.indirect_heat_integration_service(zone, args={"period_idx": 0})

    assert out is zone
    assert calls == [
        ("total_process", {"period_idx": 0}),
        ("total_site", {"period_idx": 0}),
    ]
    assert TargetType.SA.value in zone.targets
    assert TargetType.II.value in zone.targets


@pytest.mark.parametrize(
    ("service_name", "stale_target_type", "base_target_type", "compute_attr"),
    [
        (
            "direct_heat_pump_service",
            TargetType.DHP.value,
            TargetType.DI.value,
            "compute_direct_heat_pump_or_refrigeration_target",
        ),
        (
            "direct_refrigeration_service",
            TargetType.DR.value,
            TargetType.DI.value,
            "compute_direct_heat_pump_or_refrigeration_target",
        ),
        (
            "indirect_heat_pump_service",
            TargetType.IHP.value,
            TargetType.II.value,
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
            TargetType.DHP.value,
            True,
        ),
        (
            "direct_refrigeration_service",
            TargetType.DR.value,
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
        target_zone.add_target(_make_target(target_zone, TargetType.DI.value))
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
    assert TargetType.DI.value in zone.targets
    assert target_id in zone.targets


@pytest.mark.parametrize(
    ("service_name", "target_id", "expected_is_heat_pumping"),
    [
        (
            "indirect_heat_pump_service",
            TargetType.IHP.value,
            True,
        ),
        (
            "indirect_refrigeration_service",
            TargetType.IR.value,
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
        target_zone.add_target(_make_target(target_zone, TargetType.II.value))
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
    assert TargetType.II.value in zone.targets
    assert target_id in zone.targets


def test_exergy_targeting_service_auto_selects_total_site_first(monkeypatch):
    zone = _make_zone()
    zone.add_target(_make_target(zone, TargetType.II.value))
    zone.add_target(_make_target(zone, TargetType.DI.value))
    enriched = []

    monkeypatch.setattr(
        svc,
        "apply_exergy_targeting",
        lambda target: enriched.append(target.type) or target,
    )

    out = svc.exergy_targeting_service(zone)

    assert out is zone
    assert enriched == [TargetType.II.value]
    assert zone._selected_exergy_target_type == TargetType.II.value


def test_exergy_targeting_service_honours_explicit_base_target_type(monkeypatch):
    zone = _make_zone()
    zone.add_target(_make_target(zone, TargetType.II.value))
    zone.add_target(_make_target(zone, TargetType.DI.value))
    enriched = []

    monkeypatch.setattr(
        svc,
        "apply_exergy_targeting",
        lambda target: enriched.append(target.type) or target,
    )

    out = svc.exergy_targeting_service(
        zone,
        args={"base_target_type": TargetType.DI.value},
    )

    assert out is zone
    assert enriched == [TargetType.DI.value]
    assert zone._selected_exergy_target_type == TargetType.DI.value


def test_exergy_targeting_service_uses_existing_matching_state_only(monkeypatch):
    zone = _make_zone()
    zone.set_period_context({"0": 0, "peak": 1}, [1.0, 1.0], 2)
    zone.add_target(
        _make_target(zone, TargetType.II.value, period_id="0", period_idx=0)
    )
    zone.add_target(
        _make_target(zone, TargetType.DI.value, period_id="peak", period_idx=1)
    )
    enriched = []

    monkeypatch.setattr(
        svc,
        "apply_exergy_targeting",
        lambda target: enriched.append((target.type, target.period_id)) or target,
    )

    out = svc.exergy_targeting_service(zone, {"period_id": "peak"})

    assert out is zone
    assert enriched == [(TargetType.DI.value, "peak")]
    assert zone._selected_exergy_target_type == TargetType.DI.value


def test_exergy_targeting_service_explicit_target_raises_when_unavailable():
    zone = _make_zone()
    zone.add_target(_make_target(zone, TargetType.DI.value))

    with pytest.raises(RuntimeError, match="requires an existing target"):
        svc.exergy_targeting_service(zone, {"base_target_type": TargetType.II.value})


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
        lambda target_zone, args=None: _make_target(target_zone, TargetType.DI.value),
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
    zone.add_target(_make_target(zone, TargetType.DI.value))
    enriched = []

    monkeypatch.setattr(
        svc,
        "apply_exergy_targeting",
        lambda target: enriched.append(target.type) or target,
    )

    out = svc.exergy_targeting_service(zone)

    assert out is zone
    assert enriched == [TargetType.DI.value]
    assert zone._selected_exergy_target_type == TargetType.DI.value


def test_exergy_targeting_service_does_not_refresh_state_mismatched_target(
    monkeypatch,
):
    zone = _make_zone()
    zone.set_period_context({"0": 0, "peak": 1}, [1.0, 1.0], 2)
    zone.add_target(
        _make_target(zone, TargetType.II.value, period_id="0", period_idx=0)
    )

    monkeypatch.setattr(
        svc,
        "apply_exergy_targeting",
        lambda target: pytest.fail("period-mismatched target should not be enriched"),
    )

    with pytest.raises(RuntimeError, match="requires an existing target"):
        svc.exergy_targeting_service(
            zone,
            {"base_target_type": TargetType.II.value, "period_id": "peak"},
        )


def test_indirect_service_skips_none_targets(monkeypatch):
    zone = _make_zone()

    monkeypatch.setattr(
        svc,
        "indirect_heat_integration_service",
        lambda target_zone, args=None: (
            target_zone.add_target(_make_target(target_zone, TargetType.II.value))
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
    assert TargetType.II.value in zone.targets
    assert TargetType.IR.value not in zone.targets


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
                "base_target_type": TargetType.DI.value,
                "period_idx": 0,
            },
        )
        di_target = target_zone.targets.get(TargetType.DI.value) or _make_target(
            target_zone,
            TargetType.DI.value,
        )
        target_zone.add_target(di_target)
        return target_zone

    def fake_get_power_cogeneration_above_pinch(target, args=None):
        calls["cogen"] += 1
        assert args == {
            "base_target_type": TargetType.DI.value,
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
        {"base_target_type": TargetType.DI.value},
    )
    area_out = svc.area_cost_targeting_service(zone)

    assert cogeneration_out is zone
    assert area_out is zone
    assert calls["direct"] == 2
    assert calls["cogen"] == 1
    assert zone.targets[TargetType.DI.value].work_target == 12.0
    assert zone.targets[TargetType.DI.value].turbine_efficiency_target == 0.42
    assert zone._selected_cogeneration_target_type == TargetType.DI.value


def test_energy_transfer_service_bootstraps_direct_target_on_leaf(monkeypatch):
    zone = _make_zone()
    calls = {"direct": 0}

    def fake_direct_heat_integration_service(
        target_zone: Zone,
        args: dict | None = None,
    ) -> Zone:
        calls["direct"] += 1
        target_zone.add_target(_make_target(target_zone, TargetType.DI.value))
        return target_zone

    monkeypatch.setattr(
        svc,
        "direct_heat_integration_service",
        fake_direct_heat_integration_service,
    )

    out = svc.energy_transfer_analysis_service(zone)

    assert out is zone
    assert calls["direct"] == 1
    assert TargetType.ET.value in zone.targets
    assert zone.targets[TargetType.ET.value].base_target_type == TargetType.DI.value
    assert zone._selected_energy_transfer_base_target_type == TargetType.DI.value


def test_energy_transfer_service_prefers_existing_total_site_target():
    zone = _make_zone()
    zone.add_target(_make_target(zone, TargetType.II.value))
    zone.add_target(_make_target(zone, TargetType.DI.value))

    out = svc.energy_transfer_analysis_service(zone)

    assert out is zone
    assert zone.targets[TargetType.ET.value].base_target_type == TargetType.II.value
    assert zone._selected_energy_transfer_base_target_type == TargetType.II.value


def test_energy_transfer_total_site_bootstrap_solves_parent_and_child_di(
    monkeypatch,
):
    zone = _make_zone()
    subzone = Zone(name="Bleaching", type=ZoneType.P.value, config=zone.config)
    zone.add_zone(subzone)
    calls: list[tuple[str, str]] = []

    def fake_direct_heat_integration_service(
        target_zone: Zone,
        args: dict | None = None,
    ) -> Zone:
        calls.append((TargetType.DI.value, target_zone.name))
        target_zone.add_target(_make_target(target_zone, TargetType.DI.value))
        return target_zone

    def fake_indirect_heat_integration_service(
        target_zone: Zone,
        args: dict | None = None,
    ) -> Zone:
        calls.append((TargetType.II.value, target_zone.name))
        assert TargetType.DI.value in target_zone.targets
        assert TargetType.DI.value in subzone.targets
        target_zone.add_target(_make_target(target_zone, TargetType.II.value))
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
        {"base_target_type": TargetType.II.value},
    )

    assert out is zone
    assert calls == [
        (TargetType.DI.value, "Plant"),
        (TargetType.DI.value, "Bleaching"),
        (TargetType.II.value, "Plant"),
    ]
    assert zone.targets[TargetType.ET.value].base_target_type == TargetType.II.value


def test_energy_transfer_total_site_uses_one_subzone_layer_of_gccs(monkeypatch):
    zone = _make_zone()
    area = Zone(name="Area", type=ZoneType.S.value, config=zone.config)
    unit = Zone(name="Unit", type=ZoneType.P.value, config=zone.config)
    zone.add_zone(area)
    area.add_zone(unit)
    calls: list[tuple[str, str]] = []
    source_names: list[str] = []

    def fake_direct_heat_integration_service(
        target_zone: Zone,
        args: dict | None = None,
    ) -> Zone:
        calls.append((TargetType.DI.value, target_zone.name))
        target_zone.add_target(_make_target(target_zone, TargetType.DI.value))
        return target_zone

    def fake_indirect_heat_integration_service(
        target_zone: Zone,
        args: dict | None = None,
    ) -> Zone:
        calls.append((TargetType.II.value, target_zone.name))
        assert TargetType.DI.value in target_zone.targets
        assert TargetType.DI.value in area.targets
        assert TargetType.DI.value not in unit.targets
        target_zone.add_target(_make_target(target_zone, TargetType.II.value))
        return target_zone

    def fake_compute_energy_transfer_target(base_target, source_targets=None):
        for source in source_targets:
            source_names.append(source["name"])
            assert source["target"].zone_name == "Area"
        target = _make_target(zone, TargetType.ET.value)
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
        {"base_target_type": TargetType.II.value},
    )

    assert out is zone
    assert calls == [
        (TargetType.DI.value, "Plant"),
        (TargetType.DI.value, "Area"),
        (TargetType.II.value, "Plant"),
    ]
    assert source_names == ["Plant/Area"]
    assert zone.targets[TargetType.ET.value].base_target_type == TargetType.II.value


def test_cogeneration_service_prefers_total_site_before_direct_integration(
    monkeypatch,
):
    zone = _make_zone()
    zone.add_target(_make_target(zone, TargetType.II.value))
    zone.add_target(_make_target(zone, TargetType.DI.value))
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
    assert selected_targets == [TargetType.II.value]
    assert zone.targets[TargetType.II.value].work_target == 8.0
    assert zone.targets[TargetType.DI.value].work_target is None
    assert zone._selected_cogeneration_target_type == TargetType.II.value


def test_cogeneration_service_falls_back_from_total_site_to_indirect_heat_pump(
    monkeypatch,
):
    zone = _make_zone()
    refresh_order: list[str] = []
    selected_targets: list[str] = []

    def fake_indirect_heat_integration_service(target_zone: Zone, args=None) -> Zone:
        refresh_order.append(TargetType.II.value)
        return target_zone

    def fake_indirect_heat_pump_service(target_zone: Zone, args=None) -> Zone:
        refresh_order.append(TargetType.IHP.value)
        target_zone.add_target(_make_target(target_zone, TargetType.IHP.value))
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
    assert refresh_order == [TargetType.II.value, TargetType.IHP.value]
    assert selected_targets == [TargetType.IHP.value]
    assert zone._selected_cogeneration_target_type == TargetType.IHP.value


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
        refresh_order.append(TargetType.DI.value)
        target_zone.add_target(_make_target(target_zone, TargetType.DI.value))
        return target_zone

    def fake_get_power_cogeneration_above_pinch(target, args=None):
        selected_targets.append(target.type)
        return target

    monkeypatch.setattr(
        svc,
        "indirect_heat_integration_service",
        _missing(TargetType.II.value),
    )
    monkeypatch.setattr(
        svc,
        "indirect_heat_pump_service",
        _missing(TargetType.IHP.value),
    )
    monkeypatch.setattr(
        svc,
        "indirect_refrigeration_service",
        _missing(TargetType.IR.value),
    )
    monkeypatch.setattr(
        svc,
        "direct_heat_pump_service",
        _missing(TargetType.DHP.value),
    )
    monkeypatch.setattr(
        svc,
        "direct_refrigeration_service",
        _missing(TargetType.DR.value),
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
        TargetType.II.value,
        TargetType.IHP.value,
        TargetType.IR.value,
        TargetType.DHP.value,
        TargetType.DR.value,
        TargetType.DI.value,
    ]
    assert selected_targets == [TargetType.DI.value]
    assert zone._selected_cogeneration_target_type == TargetType.DI.value


@pytest.mark.parametrize(
    "base_target_type",
    [TargetType.II.value, TargetType.IHP.value, TargetType.DI.value],
)
def test_cogeneration_service_explicit_override_targets_exact_family(
    monkeypatch,
    base_target_type: str,
):
    zone = _make_zone()
    for target_type in (
        TargetType.II.value,
        TargetType.IHP.value,
        TargetType.IR.value,
        TargetType.DHP.value,
        TargetType.DR.value,
        TargetType.DI.value,
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
        svc.power_cogeneration_service(zone, {"base_target_type": TargetType.SA.value})


def test_cogeneration_service_explicit_target_raises_when_unavailable(monkeypatch):
    zone = _make_zone()

    monkeypatch.setattr(
        svc,
        "indirect_heat_integration_service",
        lambda target_zone, args=None: target_zone,
    )

    with pytest.raises(
        RuntimeError,
        match="could not produce target 'Indirect'",
    ):
        svc.power_cogeneration_service(zone, {"base_target_type": TargetType.II.value})


def test_cogeneration_service_refreshes_state_mismatched_target(monkeypatch):
    zone = _make_zone()
    zone.set_period_context({"peak": 0, "shoulder": 1}, weights=None, num_periods=2)
    zone.add_target(
        _make_target(
            zone,
            TargetType.II.value,
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
                TargetType.II.value,
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
    assert zone._selected_cogeneration_target_type == TargetType.II.value


def test_cogeneration_service_no_viable_stage_does_not_fall_back(monkeypatch):
    zone = _make_zone()
    zone.add_target(_make_target(zone, TargetType.II.value))
    zone.add_target(_make_target(zone, TargetType.DI.value))
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
    assert selected_targets == [TargetType.II.value]
    assert zone.targets[TargetType.II.value].work_target is None
    assert zone.targets[TargetType.DI.value].work_target is None
    assert zone._selected_cogeneration_target_type == TargetType.II.value


def test_direct_service_records_selected_period_id_on_zone(monkeypatch):
    zone = _make_zone()
    zone.set_period_context({"0": 0, "peak": 1}, [1.0, 1.0], 2)
    monkeypatch.setattr(
        svc,
        "compute_direct_integration_targets",
        lambda target_zone, args=None: _make_target(target_zone, TargetType.DI.value),
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
    zone.targets[TargetType.DI.value] = _make_target(
        zone,
        TargetType.DI.value,
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
            _make_target(
                target_zone, TargetType.DI.value, period_id="peak", period_idx=1
            )
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
            TargetType.DHP.value,
            period_id="peak",
            period_idx=1,
        ),
    )

    out = svc.direct_heat_pump_service(zone, {"period_id": "peak", "period_idx": 1})

    assert out is zone
    assert calls["direct"] == 1
    assert zone.targets[TargetType.DI.value].period_idx == 1
    assert zone.targets[TargetType.DHP.value].period_idx == 1


def test_indirect_heat_pump_service_refreshes_total_site_for_new_period(monkeypatch):
    zone = _make_zone()
    zone.set_period_context({"0": 0, "peak": 1}, [1.0, 1.0], 2)
    zone.targets[TargetType.II.value] = _make_target(
        zone,
        TargetType.II.value,
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
            _make_target(
                target_zone, TargetType.II.value, period_id="peak", period_idx=1
            )
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
            TargetType.IHP.value,
            period_id="peak",
            period_idx=1,
        ),
    )

    out = svc.indirect_heat_pump_service(zone, {"period_id": "peak", "period_idx": 1})

    assert out is zone
    assert calls["indirect"] == 1
    assert zone.targets[TargetType.II.value].period_idx == 1
    assert zone.targets[TargetType.IHP.value].period_idx == 1
