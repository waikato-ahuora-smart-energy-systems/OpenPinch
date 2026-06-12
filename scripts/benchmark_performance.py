"""Benchmark OpenPinch load and solve performance.

This script is intentionally manual/non-gating. It gives optimization work a
repeatable baseline without adding runtime-sensitive assertions to CI.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import tracemalloc
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from OpenPinch.classes.pinch_problem import PinchProblem
from OpenPinch.classes.problem_table import ProblemTable
from OpenPinch.classes.stream import Stream
from OpenPinch.classes.value import Value

SAMPLE_CASES = (
    "basic_pinch.json",
    "zonal_site.json",
    "zonal_site_multistate.json",
    "crude_preheat_train_multistate.json",
    "heat_pump_targeting.json",
)


@dataclass
class CounterState:
    value_init: int = 0
    stream_init: int = 0
    interval_insert: int = 0
    deepcopy: int = 0


@contextmanager
def counted_hotspots():
    counters = CounterState()
    original_value_init = Value.__init__
    original_stream_init = Stream.__init__
    original_interval_insert = ProblemTable.insert_temperature_interval
    original_deepcopy = copy.deepcopy

    def value_init(self, *args, **kwargs):
        counters.value_init += 1
        return original_value_init(self, *args, **kwargs)

    def stream_init(self, *args, **kwargs):
        counters.stream_init += 1
        return original_stream_init(self, *args, **kwargs)

    def interval_insert(self, *args, **kwargs):
        counters.interval_insert += 1
        return original_interval_insert(self, *args, **kwargs)

    def deepcopy(obj, memo=None):
        counters.deepcopy += 1
        return original_deepcopy(obj, memo)

    Value.__init__ = value_init
    Stream.__init__ = stream_init
    ProblemTable.insert_temperature_interval = interval_insert
    copy.deepcopy = deepcopy
    try:
        yield counters
    finally:
        Value.__init__ = original_value_init
        Stream.__init__ = original_stream_init
        ProblemTable.insert_temperature_interval = original_interval_insert
        copy.deepcopy = original_deepcopy


def time_case(name: str, factory: Callable[[], Any]) -> dict[str, Any]:
    with counted_hotspots() as counters:
        tracemalloc.start()
        start = time.perf_counter()
        problem = PinchProblem(factory())
        load_seconds = time.perf_counter() - start
        solve_start = time.perf_counter()
        problem.target()
        solve_seconds = time.perf_counter() - solve_start
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    return {
        "name": name,
        "load_seconds": load_seconds,
        "solve_seconds": solve_seconds,
        "total_seconds": load_seconds + solve_seconds,
        "peak_mib": peak_bytes / (1024 * 1024),
        "value_init": counters.value_init,
        "stream_init": counters.stream_init,
        "interval_insert": counters.interval_insert,
        "deepcopy": counters.deepcopy,
    }


def synthetic_case(stream_count: int, *, randomized: bool) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    streams = []
    for idx in range(stream_count):
        is_hot = idx % 2 == 0
        base = float(40 + (idx % 40) * 3)
        span = float(30 + (idx % 12) * 5)
        if randomized:
            base += float(rng.uniform(-5.0, 5.0))
            span += float(rng.uniform(-2.0, 8.0))
        t_supply = base + span if is_hot else base
        t_target = base if is_hot else base + span
        streams.append(
            {
                "zone": "Plant",
                "name": f"S{idx:05d}",
                "t_supply": {"value": t_supply, "unit": "degC"},
                "t_target": {"value": t_target, "unit": "degC"},
                "heat_flow": {"value": span * (5.0 + idx % 7), "unit": "kW"},
                "dt_cont": {"value": 5.0, "unit": "degC"},
                "htc": {"value": 1.0, "unit": "kW/m^2/degC"},
            }
        )
    return {
        "streams": streams,
        "utilities": [
            {
                "name": "HU",
                "type": "Hot",
                "t_supply": {"value": 260.0, "unit": "degC"},
                "t_target": {"value": 259.9, "unit": "degC"},
                "dt_cont": {"value": 1.0, "unit": "degC"},
                "price": {"value": 10.0, "unit": "$/MWh"},
                "htc": {"value": 1.0, "unit": "kW/m^2/degC"},
                "heat_flow": None,
            },
            {
                "name": "CU",
                "type": "Cold",
                "t_supply": {"value": 10.0, "unit": "degC"},
                "t_target": {"value": 10.1, "unit": "degC"},
                "dt_cont": {"value": 1.0, "unit": "degC"},
                "price": {"value": 1.0, "unit": "$/MWh"},
                "htc": {"value": 1.0, "unit": "kW/m^2/degC"},
                "heat_flow": None,
            },
        ],
        "options": {"DO_INDIRECT_PROCESS_TARGETING": False},
        "zone_tree": {"name": "Plant", "type": "Process Zone"},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--synthetic-sizes", type=int, nargs="*", default=[100, 1000])
    parser.add_argument("--include-10000", action="store_true")
    args = parser.parse_args()

    sample_root = Path("OpenPinch/data/sample_cases")
    results = []
    for case_name in SAMPLE_CASES:
        path = sample_root / case_name
        if path.exists():
            results.append(time_case(case_name, lambda path=path: path))

    sizes = list(args.synthetic_sizes)
    if args.include_10000 and 10000 not in sizes:
        sizes.append(10000)
    for size in sizes:
        results.append(
            time_case(
                f"synthetic_fixed_{size}",
                lambda size=size: synthetic_case(size, randomized=False),
            )
        )
        results.append(
            time_case(
                f"synthetic_random_{size}",
                lambda size=size: synthetic_case(size, randomized=True),
            )
        )

    payload = json.dumps(results, indent=2)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
