"""Contracts for the thermodynamic-only utility-placement surface."""

from __future__ import annotations

import inspect
from pathlib import Path

from OpenPinch.application._problem.accessors.target import _TargetAccessor
from OpenPinch.contracts import utility_placement as contracts


def test_public_contract_has_no_monetary_surface() -> None:
    assert not hasattr(contracts, "UtilityPlacementObjective")
    assert not hasattr(contracts, "MonetaryCostBreakdown")

    request_fields = contracts.UtilityPlacementRequest.model_fields
    template_fields = contracts.UtilityLevelTemplate.model_fields
    result_fields = contracts.UtilityPlacementResult.model_fields
    period_fields = contracts.PlacementPeriodResult.model_fields
    candidate_fields = contracts.UtilityPlacementCandidate.model_fields

    assert "objective" not in request_fields
    assert "electricity_price" not in request_fields
    assert "utility_price" not in template_fields
    assert "cogeneration_eligible" not in template_fields
    assert "objective" not in result_fields
    assert "monetary" not in period_fields
    assert "monetary_total" not in candidate_fields


def test_public_accessor_accepts_only_thermodynamic_placement_inputs() -> None:
    parameters = inspect.signature(_TargetAccessor.utility_placement).parameters

    assert "objective" not in parameters
    assert "electricity_price" not in parameters
    assert "turbine_settings" not in parameters


def test_placement_specific_monetary_modules_are_absent() -> None:
    package = Path(__file__).resolve().parents[3] / "OpenPinch" / "analysis" / "utility_placement"

    assert not (package / "economics.py").exists()
    assert not (package / "cogeneration.py").exists()


def test_obsolete_placement_presentation_module_is_absent() -> None:
    presentation = Path(__file__).resolve().parents[3] / "OpenPinch" / "presentation"

    assert not (presentation / "utility_placement.py").exists()
