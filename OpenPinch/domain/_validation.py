"""Reusable pydantic field validators for domain and contract models.

Each factory returns a validator object for use as ``Annotated`` metadata,
for example ``Annotated[str, non_empty_str("name must not be empty")]``.
Raising ``ValueError`` inside these validators produces the same pydantic
``value_error`` (``"Value error, <message>"``) as an equivalent
``field_validator``.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import AfterValidator, BeforeValidator

__all__ = ["finite_float", "non_empty_str"]


def non_empty_str(message: str) -> AfterValidator:
    """Return an after-validator that strips text and rejects blank values.

    The stripped text is the validated field value.
    """

    def _validate(value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError(message)
        return text

    return AfterValidator(_validate)


def finite_float(
    message: str = "value must be finite",
    *,
    ge: float | None = None,
    gt: float | None = None,
    bound_message: str | None = None,
    mode: Literal["after", "before"] = "after",
) -> AfterValidator | BeforeValidator:
    """Return a validator that coerces to a finite ``float`` with optional bounds.

    ``message`` is raised for non-finite values; ``bound_message`` (defaulting
    to ``message``) is raised when ``ge``/``gt`` bounds are violated. Negative
    zero is normalised to ``0.0``. With ``mode="before"`` the check runs ahead
    of pydantic's own float parsing, so the raw input is passed to ``float()``.
    """
    if ge is not None and gt is not None:
        raise ValueError("finite_float accepts at most one of ge and gt")
    bound_text = message if bound_message is None else bound_message

    def _validate(value: object) -> float:
        result = float(value)  # type: ignore[arg-type]
        if not math.isfinite(result):
            raise ValueError(message)
        if ge is not None and result < ge:
            raise ValueError(bound_text)
        if gt is not None and result <= gt:
            raise ValueError(bound_text)
        return 0.0 if result == 0.0 else result

    if mode == "before":
        return BeforeValidator(_validate)
    return AfterValidator(_validate)
