"""Tests for the reusable Annotated field validators."""

from typing import Annotated

import pytest
from pydantic import BaseModel, ValidationError

from OpenPinch.domain._validation import finite_float, non_empty_str


class _Model(BaseModel):
    name: Annotated[str, non_empty_str("name must not be empty")] = "x"
    after: Annotated[float, finite_float(ge=0.0, bound_message="after < 0")] = 0.0
    before: Annotated[float, finite_float(gt=0.0, mode="before")] = 1.0


def test_non_empty_str_strips_and_rejects_blank_text():
    assert _Model(name="  pump  ").name == "pump"
    with pytest.raises(ValidationError) as excinfo:
        _Model(name="   ")
    (error,) = excinfo.value.errors()
    assert error["type"] == "value_error"
    assert error["loc"] == ("name",)
    assert error["msg"] == "Value error, name must not be empty"


def test_finite_float_bounds_and_negative_zero():
    assert _Model(after=-0.0).after == 0.0
    assert str(_Model(after=-0.0).after) == "0.0"
    with pytest.raises(ValidationError, match="after < 0"):
        _Model(after=-1.0)
    with pytest.raises(ValidationError, match="value must be finite"):
        _Model(before=0.0)


def test_finite_float_before_mode_rejects_non_finite_input():
    assert _Model(before="2.5").before == 2.5
    with pytest.raises(ValidationError, match="value must be finite"):
        _Model(before=float("inf"))


def test_finite_float_rejects_both_bounds():
    with pytest.raises(ValueError, match="at most one"):
        finite_float(ge=0.0, gt=0.0)
