"""Tests for lazy public service package entry points."""

from __future__ import annotations

from types import SimpleNamespace

import OpenPinch.services as services
import OpenPinch.services.heat_exchanger_network_controllability as controllability
from OpenPinch.services import services_entry


def test_public_services_lazy_loader_returns_services_entry_module():
    services._load_services_entry_module.cache_clear()

    assert services._load_services_entry_module() is services_entry


def test_public_service_wrappers_delegate_to_lazy_services_entry(monkeypatch):
    calls = []

    def recorder(name):
        def wrapped(*args, **kwargs):
            calls.append((name, args, kwargs))
            return f"{name}-result"

        return wrapped

    stub_module = SimpleNamespace(
        data_preprocessing_service=recorder("data_preprocessing_service"),
        direct_heat_integration_service=recorder("direct_heat_integration_service"),
        exergy_targeting_service=recorder("exergy_targeting_service"),
        indirect_heat_integration_service=recorder("indirect_heat_integration_service"),
        direct_heat_pump_service=recorder("direct_heat_pump_service"),
        indirect_heat_pump_service=recorder("indirect_heat_pump_service"),
        direct_refrigeration_service=recorder("direct_refrigeration_service"),
        indirect_refrigeration_service=recorder("indirect_refrigeration_service"),
        power_cogeneration_service=recorder("power_cogeneration_service"),
        area_cost_targeting_service=recorder("area_cost_targeting_service"),
        energy_transfer_analysis_service=recorder("energy_transfer_analysis_service"),
    )
    monkeypatch.setattr(services, "_load_services_entry_module", lambda: stub_module)

    input_data = object()
    zone = object()
    args = {"period_id": "peak"}

    assert services.data_preprocessing_service(input_data, "Plant") == (
        "data_preprocessing_service-result"
    )
    assert services.direct_heat_integration_service(zone, args) == (
        "direct_heat_integration_service-result"
    )
    assert (
        services.exergy_targeting_service(zone, args)
        == "exergy_targeting_service-result"
    )
    assert services.indirect_heat_integration_service(zone, args) == (
        "indirect_heat_integration_service-result"
    )
    assert (
        services.direct_heat_pump_service(zone, args)
        == "direct_heat_pump_service-result"
    )
    assert services.indirect_heat_pump_service(zone, args) == (
        "indirect_heat_pump_service-result"
    )
    assert services.direct_refrigeration_service(zone, args) == (
        "direct_refrigeration_service-result"
    )
    assert services.indirect_refrigeration_service(zone, args) == (
        "indirect_refrigeration_service-result"
    )
    assert services.power_cogeneration_service(zone, args) == (
        "power_cogeneration_service-result"
    )
    assert services.area_cost_targeting_service(zone, args) == (
        "area_cost_targeting_service-result"
    )
    assert services.energy_transfer_analysis_service(zone, args) == (
        "energy_transfer_analysis_service-result"
    )

    assert calls[0] == (
        "data_preprocessing_service",
        (),
        {"input_data": input_data, "project_name": "Plant"},
    )
    assert calls[1:] == [
        ("direct_heat_integration_service", (zone, args), {}),
        ("exergy_targeting_service", (zone, args), {}),
        ("indirect_heat_integration_service", (zone, args), {}),
        ("direct_heat_pump_service", (zone, args), {}),
        ("indirect_heat_pump_service", (zone, args), {}),
        ("direct_refrigeration_service", (zone, args), {}),
        ("indirect_refrigeration_service", (zone, args), {}),
        ("power_cogeneration_service", (zone, args), {}),
        ("area_cost_targeting_service", (zone, args), {}),
        ("energy_transfer_analysis_service", (zone, args), {}),
    ]


def test_public_controllability_service_delegates_to_package_function(monkeypatch):
    calls = []

    def fake_quantify(network, **kwargs):
        calls.append((network, kwargs))
        return {"ok": True}

    monkeypatch.setattr(
        controllability,
        "quantify_heat_exchanger_network_controllability",
        fake_quantify,
    )

    network = object()

    assert services.heat_exchanger_network_controllability_service(
        network,
        perturbation=0.1,
    ) == {"ok": True}
    assert calls == [(network, {"perturbation": 0.1})]
