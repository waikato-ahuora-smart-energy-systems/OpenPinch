from __future__ import annotations

from pathlib import Path

import pytest

import OpenPinch.resources as resource_module
from OpenPinch.analysis.heat_pumps.performance_maps.resources import (
    load_tespy_compressor_characteristic,
    read_tespy_compressor_characteristic_bytes,
)
from OpenPinch.contracts.hpr_performance_map import HprPerformanceMap
from OpenPinch.resources import (
    copy_sample_case,
    list_hpr_performance_map_contract_resources,
    list_sample_cases,
    load_hpr_performance_map_contract_resource,
    read_hpr_performance_map_contract_resource,
)

EXPECTED_HPR_CONTRACT_RESOURCES = [
    "heat-pump-1.0.json",
    "refrigeration-1.0.json",
    "schema-1.0.json",
]
EXPECTED_TESPY_CHARACTERISTIC_SHA256 = (
    "f7e1864476a243366aac8721a41aa09e4b909025414428d59b4ae7df9683b8af"
)


def test_copy_sample_case_uses_original_name_for_directory_destinations(
    tmp_path: Path,
):
    sample_name = list_sample_cases()[0]
    destination = tmp_path / "samples"
    destination.mkdir()

    copied = copy_sample_case(sample_name, destination)

    assert copied == destination / sample_name
    assert copied.exists()


def test_hpr_performance_map_contract_resource_catalog_is_closed() -> None:
    assert (
        list_hpr_performance_map_contract_resources() == EXPECTED_HPR_CONTRACT_RESOURCES
    )


@pytest.mark.parametrize(
    "name",
    ["heat-pump-1.0.json", "refrigeration-1.0.json"],
)
def test_hpr_golden_fixture_resource_is_readable_and_valid(name: str) -> None:
    text = read_hpr_performance_map_contract_resource(name)
    payload = load_hpr_performance_map_contract_resource(name)

    assert text.endswith("\n")
    assert HprPerformanceMap.model_validate(payload).schema_version == "1.0"


def test_loaded_hpr_contract_resource_is_detached() -> None:
    first = load_hpr_performance_map_contract_resource("heat-pump-1.0.json")
    first["map_id"] = "changed"
    first["points"].clear()

    second = load_hpr_performance_map_contract_resource("heat-pump-1.0.json")
    assert second["map_id"] == "openpinch-golden-heat-pump-1.0"
    assert len(second["points"]) == 3


def test_unknown_hpr_contract_resource_fails_closed() -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Unknown OpenPinch HPR performance-map contract resource",
    ):
        read_hpr_performance_map_contract_resource("future-2.0.json")


def test_tespy_characteristic_package_resource_loads_with_pinned_identity() -> None:
    content = read_tespy_compressor_characteristic_bytes()
    characteristic = load_tespy_compressor_characteristic()

    assert len(content) == 458
    assert characteristic.characteristic_set_id == (
        "openpinch-single-stage-compressor-v1"
    )
    assert characteristic.sha256 == EXPECTED_TESPY_CHARACTERISTIC_SHA256
    assert len(characteristic.points) == 17


def test_non_object_hpr_contract_resource_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resource_name = "not-an-object.json"
    (tmp_path / resource_name).write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(
        resource_module,
        "_HPR_PERFORMANCE_MAP_CONTRACT_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        resource_module,
        "_HPR_PERFORMANCE_MAP_CONTRACT_RESOURCES",
        (resource_name,),
    )

    with pytest.raises(ValueError, match="is not an object"):
        load_hpr_performance_map_contract_resource(resource_name)
