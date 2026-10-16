"""Display text for the editable configuration fields.

Each field has a label, the unit its value is given in (``None`` when it has
none) and a one-sentence description, for option editors.
"""

from __future__ import annotations

from typing import NamedTuple


class FieldText(NamedTuple):
    label: str
    unit: str | None
    description: str


# fmt: off
CONFIGURATION_FIELD_TEXT: dict[str, FieldText] = {
    "THERMAL_DT_CONT": FieldText(
        "Minimum contribution temperature difference", "delta_degC",
        "The dt_cont of any stream or utility that does not give its own.",
    ),
}
# fmt: on
