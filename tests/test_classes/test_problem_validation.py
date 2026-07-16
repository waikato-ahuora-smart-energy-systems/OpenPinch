"""Validation edge-case tests for problem input reporting."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from OpenPinch.classes._pinch_problem.input import semantics, validation
from OpenPinch.classes.value import Value
from OpenPinch.lib.schemas.io import TargetInput

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / ("problem_validation_cases.json")
)


@pytest.fixture(scope="module")
def validation_cases() -> dict[str, dict]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_validate_problem_inputs_reports_schema_semantic_and_success_cases(
    validation_cases,
):
    valid = validation.validate_problem_inputs(validation_cases["valid_problem"])
    assert isinstance(valid, TargetInput)

    with pytest.raises(ValueError, match="At least one stream"):
        validation.validate_problem_inputs(validation_cases["empty_stream_problem"])

    with pytest.raises(ValueError, match="Input validation failed"):
        validation.validate_problem_inputs(validation_cases["schema_invalid_problem"])


def test_validation_report_and_warning_semantics(validation_cases):
    assert (
        validation.build_validation_context("not a mapping", source_kind="json") == {}
    )

    report = validation.build_validation_report(validation_cases["warning_problem"])
    assert report.valid is True
    assert [issue.severity for issue in report.issues] == ["warning", "warning"]

    validated = TargetInput.model_validate(validation_cases["warning_problem"])
    with pytest.warns(UserWarning, match="warning"):
        validation.validate_problem_semantics(validated, context={})


def test_validation_report_rejects_discontinuous_segmented_utility_before_loading():
    payload = {
        "streams": [
            {
                "zone": "Site",
                "name": "C1",
                "t_supply": 20.0,
                "t_target": 80.0,
                "heat_flow": 60.0,
            }
        ],
        "utilities": [
            {
                "name": "Steam",
                "type": "Hot",
                "segments": [
                    {
                        "t_supply": 300.0,
                        "t_target": 250.0,
                        "heat_flow": 50.0,
                    },
                    {
                        "t_supply": 240.0,
                        "t_target": 200.0,
                        "heat_flow": 40.0,
                    },
                ],
            }
        ],
    }

    report = validation.build_validation_report(payload)

    assert report.valid is False
    assert any(
        issue.path == "utilities[0].segments[1].t_supply"
        and "previous segment" in issue.message
        for issue in report.issues
    )
    with pytest.raises(
        ValueError,
        match=r"utilities\[0\]\.segments\[1\]\.t_supply",
    ):
        validation.validate_problem_inputs(payload)


def test_validation_report_rejects_segmented_parent_aggregate_mismatch():
    payload = {
        "streams": [
            {
                "zone": "Site",
                "name": "H1",
                "t_supply": 200.0,
                "t_target": 100.0,
                "heat_flow": 999.0,
                "segments": [
                    {
                        "t_supply": 200.0,
                        "t_target": 150.0,
                        "heat_flow": 50.0,
                    },
                    {
                        "t_supply": 150.0,
                        "t_target": 100.0,
                        "heat_flow": 100.0,
                    },
                ],
            }
        ],
        "utilities": [],
    }

    report = validation.build_validation_report(payload)

    assert report.valid is False
    assert any(
        issue.path == "streams[0].heat_flow"
        and "authoritative profile" in issue.message
        for issue in report.issues
    )
    with pytest.raises(ValueError, match=r"streams\[0\]\.heat_flow"):
        validation.validate_problem_inputs(payload)


def test_schema_error_formatting_and_record_context_helpers():
    context = {
        "streams": [
            {"name": "", "row": 4, "sheet": "Streams"},
        ],
        "utilities": [
            {"name": "Steam", "entry": "utilities[0]"},
        ],
    }

    assert "Field 'options.bad'" in validation.format_single_validation_error(
        {"loc": ("options", "bad"), "msg": "Bad option"},
        problem_data={},
        context=context,
    )
    assert validation.path_from_loc((0, "name")) == "[0].name"
    assert validation.validation_record_label(None, None, context) is None
    assert validation.validation_record_label("streams", 0, context) == "Stream 1"
    assert validation.lookup_record_context({}, "streams", 3) == {
        "index": 3,
        "section": "streams",
    }
    assert validation.describe_record("streams", 0, context["streams"][0]) == (
        "Stream 1 (Streams row 4)"
    )
    assert validation.describe_record("utilities", 0, context["utilities"][0]) == (
        "Utility 1 'Steam' (entry utilities[0])"
    )
    schema_report = validation.build_validation_report(
        {"streams": "not a list", "utilities": []}
    )
    assert schema_report.valid is False
    assert schema_report.issues[0].severity == "error"
    assert (
        validation.build_validation_context(
            {"streams": "not a list"},
            source_kind="json",
        )
        == {}
    )
    issue = validation.schema_issue_to_view(
        {"loc": ("streams", 2, "t_supply"), "msg": "Missing field"},
        context={"streams": [{"name": "H1"}]},
    )
    assert issue.path == "streams[2].t_supply"
    assert issue.field == "t_supply"
    assert issue.record_label == "Stream 3"
    assert validation.path_from_loc(("streams", 0, "name")) == "streams[0].name"
    assert validation._build_record_context(
        "streams",
        0,
        {"name": "H1", "zone": "Site"},
        source_kind="excel",
    ) == {
        "index": 0,
        "section": "streams",
        "name": "H1",
        "zone": "Site",
        "sheet": "Stream Data",
        "row": 3,
    }
    assert (
        validation._build_record_context(
            "utilities",
            0,
            {"name": "Steam"},
            source_kind="json",
        )["entry"]
        == 1
    )


def test_value_coercion_and_missing_raw_value_edges():
    record = SimpleNamespace(t_supply={"value": 10.0, "unit": "not-a-unit"})
    values, issues = semantics._coerce_validation_values(
        record,
        section="streams",
        record_index=0,
        record_label="Stream 1",
        field_names=("t_supply",),
        optional_field_names=(),
        config={},
    )

    assert values["t_supply"] is None
    assert issues[0].path == "streams[0].t_supply"

    class DumpValue:
        def model_dump(self, *, mode):
            assert mode == "python"
            return {"values": [None, None]}

    assert semantics._raw_value_is_missing(None) is True
    assert semantics._raw_value_is_missing(DumpValue()) is True
    assert semantics._raw_value_is_missing({"value": None}) is True
    assert semantics._raw_value_is_missing({"values": None}) is True
    assert semantics._raw_value_is_missing({"values": [None, None]}) is True
    assert semantics._raw_value_is_missing({}) is False
    assert semantics._raw_value_is_missing(12.0) is False

    optional_values, optional_issues = semantics._coerce_validation_values(
        SimpleNamespace(dt_cont={"value": None}),
        section="streams",
        record_index=0,
        record_label="Stream 1",
        field_names=("dt_cont",),
        optional_field_names=("dt_cont",),
        config={},
    )
    assert optional_values["dt_cont"] is None
    assert optional_issues == []


def test_stream_value_state_validation_edges():
    values = {
        "t_supply": Value([None, float("nan"), 100.0], "degC"),
        "t_target": Value([10.0, 20.0, 100.0], "degC"),
        "heat_flow": Value([1.0, 2.0, 3.0], "kW"),
        "dt_cont": Value([1.0, 2.0, 3.0], "delta_degC"),
        "htc": Value([1.0, 2.0, 3.0], "kW/m^2/K"),
    }

    issues = semantics._validate_stream_record_states(
        values,
        section="streams",
        record_index=0,
        record_label="Stream 1",
    )

    assert any("must differ" in issue.message for issue in issues)
    assert (
        semantics._validate_stream_record_states(
            {"t_supply": None, "t_target": values["t_target"]},
            section="streams",
            record_index=0,
            record_label="Stream 1",
        )
        == []
    )


def test_stream_classification_and_utility_state_edges():
    class PeriodValues:
        def __init__(self, values):
            self.period_values = values

        def __getitem__(self, index):
            return self.period_values[index]

    values = {
        "t_supply": PeriodValues([None, float("nan"), 110.0, 20.0]),
        "t_target": PeriodValues([10.0, 20.0, 50.0, 80.0]),
        "heat_flow": Value([1.0, 1.0, 1.0, 1.0], "kW"),
        "dt_cont": Value([1.0, 1.0, 1.0, 1.0], "delta_degC"),
        "htc": Value([1.0, 1.0, 1.0, 1.0], "kW/m^2/K"),
    }

    issues = semantics._validate_stream_record_states(
        values,
        section="streams",
        record_index=0,
        record_label="Stream 1",
    )
    utility_issues = semantics._validate_utility_record_states(
        {
            "t_supply": Value([180.0, 181.0], "degC"),
            "t_target": Value([120.0, 121.0], "degC"),
            "dt_cont": Value([1.0, 1.0], "delta_degC"),
            "price": Value([1.0, 1.0], "USD/kWh"),
            "heat_flow": Value([1.0, 1.0], "kW"),
            "htc": Value([1.0, 1.0], "kW/m^2/K"),
        },
        section="utilities",
        record_index=0,
        record_label="Utility 1",
    )

    assert any("classify consistently" in issue.message for issue in issues)
    assert utility_issues == []


def test_finiteness_non_negative_and_period_suffix_helpers():
    class PeriodValues:
        def __init__(self, values):
            self.period_values = values

        def __getitem__(self, index):
            return self.period_values[index]

    finite_issues = semantics._validate_value_finiteness(
        PeriodValues([None, float("inf"), 1.0]),
        section="streams",
        record_index=0,
        record_label="Stream 1",
        field_name="heat_flow",
    )
    non_negative_issues = semantics._validate_non_negative_states(
        PeriodValues([None, float("inf"), -1.0, 0.0]),
        section="streams",
        record_index=0,
        record_label="Stream 1",
        field_name="heat_flow",
        severity="error",
        message="Value must be non-negative.",
    )

    assert finite_issues[0].message == "Value must be finite for period_id '1'."
    assert non_negative_issues[0].message == (
        "Value must be non-negative for period_id '2'."
    )
    assert semantics._period_suffix(None) == ""
    assert semantics._with_period_suffix("No period.", None) == "No period."
