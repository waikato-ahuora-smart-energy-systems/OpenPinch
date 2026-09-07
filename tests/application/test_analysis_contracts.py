"""Cross-family analysis metadata and extension boundary contracts."""

from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from OpenPinch import PinchProblem
from tests.application.test_hpr_period_batch_boundaries import _two_period_payload


def test_result_provenance_is_immutable_and_round_trips():
    from OpenPinch.domain.analysis import AnalysisProvenance

    p = PinchProblem(_two_period_payload(), project_name="Site")
    result = p.target.direct_heat_integration(zone="AreaA", period_id="base")
    provenance = result.provenance
    assert provenance.method_id == "target.direct_heat_integration"
    assert provenance.zone_address == "Site/AreaA"
    assert provenance.period_ids == ("base",)
    assert len(provenance.input_fingerprint) == 64
    assert (
        AnalysisProvenance.model_validate_json(provenance.model_dump_json())
        == provenance
    )
    with pytest.raises((ValidationError, FrozenInstanceError)):
        provenance.method_id = "changed"
    assert p.results.targets[0].provenance == provenance


def test_foreign_and_stale_base_targets_are_rejected():
    p = PinchProblem(_two_period_payload(), project_name="Site")
    q = PinchProblem(_two_period_payload(), project_name="Site")
    original = p.target.direct_heat_integration(zone="AreaA", period_id="base")
    q.target.direct_heat_integration(zone="AreaA", period_id="base")
    with pytest.raises(ValueError, match="base_target"):
        q.target.exergy(zone="AreaA", period_id="base", base_target=original)
    p.update_options({"ENV_TEMPERATURE": 80.0})
    p.target.direct_heat_integration(zone="AreaA", period_id="base")
    with pytest.raises(ValueError, match="base_target"):
        p.target.exergy(zone="AreaA", period_id="base", base_target=original)


def test_catalog_matches_live_target_and_design_methods():
    from OpenPinch.application._problem.targeting.catalog import METHOD_CATALOG

    p = PinchProblem()
    for prefix in ("target", "design"):
        accessor = getattr(p, prefix)
        actual = {
            name
            for name in dir(accessor)
            if not name.startswith("_") and callable(getattr(accessor, name))
        }
        described = {
            key.removeprefix(prefix + ".")
            for key in METHOD_CATALOG
            if key.startswith(prefix + ".")
        }
        assert described == actual
    assert not METHOD_CATALOG["target.brayton_heat_pump"].available
    with pytest.raises(TypeError):
        METHOD_CATALOG["new"] = None


def test_nonthermal_result_has_family_report_adapter():
    from OpenPinch.presentation.reporting.adapters import adapt_analysis_result

    p = PinchProblem(_two_period_payload(), project_name="Site")
    result = p.target.heat_recovery_dt_min(
        heat_recovery=60.0, zone="AreaA", period_id="base"
    )
    report = adapt_analysis_result(result)
    assert report.provenance.method_id == "target.heat_recovery_dt_min"
    assert "Qh" not in type(report).model_fields
    assert p.results is None


def test_catalog_workspace_period_adapters_and_docs_remain_consistent():
    from OpenPinch.application._problem.accessors.target import (
        _AllPeriodsTargetAccessor,
    )
    from OpenPinch.application._problem.targeting.catalog import METHOD_CATALOG
    from OpenPinch.application.workspace import (
        _CaseBatchAllPeriodsTargetAccessor,
        _CaseBatchDesignAccessor,
        _CaseBatchTargetAccessor,
    )
    from OpenPinch.domain.configuration_fields import USER_CONFIG_FIELD_SPECS
    from OpenPinch.presentation.reporting.adapters import RESULT_ADAPTERS
    from scripts.generate_analysis_catalog import DESTINATION, render_catalog

    def names(owner):
        return {
            name
            for name in dir(owner)
            if not name.startswith("_") and callable(getattr(owner, name))
        }

    targets = {
        name.split(".")[1] for name in METHOD_CATALOG if name.startswith("target.")
    }
    designs = {
        name.split(".")[1] for name in METHOD_CATALOG if name.startswith("design.")
    }
    assert names(_CaseBatchTargetAccessor) == targets - {
        "hpr_performance_map",
    }
    assert names(_CaseBatchDesignAccessor) == designs
    period_names = targets - {
        "hpr_performance_map",
        "brayton_heat_pump",
        "brayton_refrigeration",
    }
    assert names(_AllPeriodsTargetAccessor) == period_names
    assert names(_CaseBatchAllPeriodsTargetAccessor) == period_names
    adapter_families = {
        "thermal",
        "inverse_dt_min",
        "placement",
        "hpr_map",
        "hen",
        "residual",
    }
    for spec in METHOD_CATALOG.values():
        assert spec.result_adapter in adapter_families
        assert set(spec.configuration_fields) <= USER_CONFIG_FIELD_SPECS.keys()
    assert RESULT_ADAPTERS
    assert DESTINATION.read_text() == render_catalog()


def test_temporary_configuration_does_not_change_prepared_input_identity():
    from OpenPinch.application._problem.arguments import temporary_zone_configuration
    from OpenPinch.application._problem.targeting.provenance import input_fingerprint

    problem = PinchProblem(_two_period_payload(), project_name="Site")
    original = input_fingerprint(problem)
    with temporary_zone_configuration(
        problem._master_zone, {"HENS_DERIVATIVE_THRESHOLDS": [9.9]}
    ):
        assert input_fingerprint(problem) == original
    problem.update_options({"ENV_TEMPERATURE": 80.0})
    assert input_fingerprint(problem) != original


def test_multiperiod_design_keeps_its_named_method_identity(monkeypatch):
    from tests.analysis.heat_exchanger_networks.test_design_workflow import (
        _public_example_problem,
        _use_fake_default_executor,
    )

    _use_fake_default_executor(monkeypatch)
    problem = _public_example_problem()
    problem.target.all_periods.direct_heat_integration()
    view = problem.design.multiperiod_heat_exchanger_network()
    assert (
        view.result.provenance.method_id == "design.multiperiod_heat_exchanger_network"
    )
    assert problem.results.design.provenance == view.result.provenance
