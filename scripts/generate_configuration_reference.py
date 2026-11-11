"""Generate the configuration options reference from the live option metadata."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from OpenPinch.presentation.configuration import configuration_options

DESTINATION = (
    Path(__file__).resolve().parents[1] / "docs/_data/configuration-options.inc"
)

GROUP_TITLES = {
    "problem": "Problem",
    "input_units": "Input Units",
    "output_units": "Output Units",
    "reporting": "Reporting",
    "environment": "Environment",
    "thermal": "Thermal",
    "direct": "Direct Integration",
    "costing": "Costing",
    "hens": "Heat Exchanger Network Synthesis",
    "hpr": "Heat Pump and Refrigeration",
    "orc": "Organic Rankine Cycle",
    "process_mvr": "Process MVR",
    "power": "Power and Cogeneration",
}


def _default(value: object) -> str:
    if isinstance(value, str) and value:
        return f"``{value}``"
    if value is None or isinstance(value, bool):
        return f"``{value}``"
    return f"``{json.dumps(value)}``"


def _limits(field) -> str:
    if field.enum_choices:
        return ", ".join(field.enum_choices)
    low, high = field.numeric_min, field.numeric_max
    if low is None and high is None:
        return ""
    if high is None:
        return f"{'>' if field.numeric_min_exclusive else '>='} {low:g}"
    if low is None:
        return f"<= {high:g}"
    return f"{'> ' if field.numeric_min_exclusive else ''}{low:g} to {high:g}"


def render_reference() -> str:
    groups: dict[str, list] = {}
    for field in configuration_options():
        groups.setdefault(field.group, []).append(field)

    out = io.StringIO()
    for group, fields in groups.items():
        title = GROUP_TITLES[group]
        out.write(f"{title}\n{'~' * len(title)}\n\n")
        out.write(".. csv-table::\n")
        out.write('   :header: "Option", "Type", "Default", "Unit", "Limits", ')
        out.write('"Description"\n')
        out.write("   :widths: 22, 8, 12, 8, 10, 40\n\n")
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, quoting=csv.QUOTE_ALL, lineterminator="\n")
        for field in fields:
            writer.writerow(
                [
                    f"``{field.name}``",
                    field.field_type,
                    _default(field.default),
                    field.unit or "",
                    _limits(field),
                    field.description,
                ]
            )
        for line in stream.getvalue().splitlines():
            out.write(f"   {line}\n")
        out.write("\n")
    return out.getvalue()


if __name__ == "__main__":
    DESTINATION.write_text(render_reference(), encoding="utf-8")
