"""CSV presentation for domain stream collections."""

from __future__ import annotations

import csv
from pathlib import Path

from ...domain.stream_collection import StreamCollection


def export_stream_collection(
    streams: StreamCollection,
    filename: str = "heat pump streams",
    *,
    output_dir: Path | None = None,
) -> Path:
    """Write ordered stream data as CSV and return the resulting path."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[3] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{filename}.csv"

    ordered_streams = list(streams)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "name",
                "t_supply",
                "t_target",
                "heat_flow",
                "dt_cont",
                "dt_cont_multiplier",
                "htc",
            ]
        )
        for stream in ordered_streams:
            writer.writerow(
                [
                    stream.name,
                    stream.t_supply,
                    stream.t_target,
                    stream.heat_flow,
                    stream.dt_cont,
                    stream.dt_cont_multiplier,
                    stream.htc,
                ]
            )
    return output_path


__all__ = ["export_stream_collection"]
