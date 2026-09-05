# Performance Test Instructions

## Purpose

Validate bounded map generation, exact candidate caching, repeat-call resource
stability, and the real public TESPy target-to-map profile. Network throughput
and concurrent-user load are N/A because OpenPinch is a local library.

## Performance Requirements

- A 10,000-point fake map completes within 5 seconds and 256 MiB traced Python
  memory.
- Candidate handling is linear in callback count, with TESPy solves no greater
  than exact cache misses.
- The call-local LRU retains at most 512 values and adds less than 64 MiB traced
  Python memory for maximum-size fake results.
- Ten fake calls and at least three real guarded calls retain no engine object
  after cleanup.
- The public real TESPy target plus minimal map completes within 300 seconds.
- Execution inside one call is sequential; independent callers may use
  process-level parallelism.

## Run Performance and Lifecycle Tests

```bash
uv run pytest --hypothesis-seed=20260715 -q \
  tests/analysis/heat_pumps/test_hpr_map_generation_properties.py \
  tests/analysis/heat_pumps/test_hpr_target_candidate_cache.py \
  tests/analysis/heat_pumps/test_hpr_simulator_stateful.py
```

Run the guarded real profile:

```bash
timeout 300s uv run pytest -q \
  tests/analysis/heat_pumps/test_hpr_tespy_target_evaluator.py::test_real_public_tespy_target_and_minimal_map_stay_within_smoke_budget
```

## Analyze a Failure

Rerun the isolated assertion on an otherwise idle machine. Distinguish wall-
clock noise from a repeatable regression, then inspect cache counters, traced
memory, retained weak references, and recorded target/map elapsed properties.
Preserve exact keys, deterministic order, physical validation, and cleanup
semantics when optimizing.
