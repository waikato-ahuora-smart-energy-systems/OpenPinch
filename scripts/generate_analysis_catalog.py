"""Generate the capability table directly from the immutable method catalog."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from OpenPinch.application._problem.targeting.catalog import METHOD_CATALOG

DESTINATION = Path(__file__).resolve().parents[1] / "docs/_data/analysis-methods.csv"


def render_catalog() -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        ["Method", "Scopes", "Periods", "Prerequisites", "Adapter", "Availability"]
    )
    for name, spec in METHOD_CATALOG.items():
        writer.writerow(
            [
                name,
                "; ".join(dict.fromkeys(spec.scopes)),
                spec.period_behavior,
                "; ".join(spec.prerequisites) or "none",
                spec.result_adapter,
                "available" if spec.available else "unavailable",
            ]
        )
    return stream.getvalue()


if __name__ == "__main__":
    DESTINATION.write_text(render_catalog(), encoding="utf-8")
