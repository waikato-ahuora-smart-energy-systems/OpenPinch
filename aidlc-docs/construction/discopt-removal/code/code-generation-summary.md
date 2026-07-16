# Experimental Discopt Integration Removal Code Generation Summary

## Outcome

The experimental Discopt integration has been removed from active package code,
HEN model contracts, scripts, tests, strategies, and developer documentation.
Couenne, IPOPT, and APOPT code paths were restored to their pre-experiment
content. Historical AI-DLC records and ignored benchmark results were preserved.

## Restored Tracked Files

These files were restored byte-for-byte to their tracked pre-Discopt content and
therefore no longer appear as modified in the Git diff:

- `OpenPinch/services/heat_exchanger_network_synthesis/common/solver/backend.py`
- `OpenPinch/services/heat_exchanger_network_synthesis/unit_models/base.py`
- `OpenPinch/services/heat_exchanger_network_synthesis/unit_models/pinch_design.py`
- `OpenPinch/services/heat_exchanger_network_synthesis/unit_models/stagewise.py`
- `tests/heat_exchanger_network_synthesis/test_pinch_design_method.py`
- `docs/developer/synthesis-dependency-policy.rst`

## Removed Experimental Files

- `OpenPinch/services/heat_exchanger_network_synthesis/common/solver/_discopt.py`
- `scripts/benchmark_hen_solvers.py`
- `tests/heat_exchanger_network_synthesis/test_benchmark_hen_solvers.py`
- `tests/strategies/hen_benchmarks.py`

These files were untracked experimental additions, so their removal leaves no
tracked deletion in the Git diff.

## Preserved Solver-Neutral Benchmark Improvements

- `scripts/benchmark_performance.py` retains actual returned-backend tracing,
  explicit verification metadata, active exchanger counts, and recovery/hot-
  utility/cold-utility duties.
- `tests/heat_exchanger_network_synthesis/test_benchmark_performance.py` retains
  the returned-backend tracer regression using supported APOPT instead of
  Discopt.
- Discopt-only termination, native-time, bound, and node telemetry was removed.

## Static Verification

- Parsed all five Python files remaining modified in the working tree.
- Found no active Discopt references under `OpenPinch/`, `scripts/`, `tests/`,
  `docs/developer/`, `pyproject.toml`, or `uv.lock`.
- Confirmed no deleted experimental or duplicate replacement files remain.
- Confirmed package metadata hashes are unchanged:
  - `pyproject.toml`: `722fc7975d2a6c120e0126d446285045655fd1e369dfcbbdd3e16383f270e859`
  - `uv.lock`: `20842cfc4883efd175c32225a626082ac0801e030dcdea20f3a408ed19914497`
- `git diff --check` passed.
- Scoped diff inspection found only the intentionally preserved solver-neutral
  benchmark changes; unrelated heat-pump and documentation work was untouched.
- The first AST command could not initialize uv's sandboxed user cache. The same
  read-only AST validation then passed with the system Python interpreter.

## Deferred Build and Test Work

- Focused HEN backend and benchmark-performance tests.
- Ruff formatting and lint checks.
- CI-selected non-solver suite with the 95% coverage gate.
- Solver-marked HEN tests.
- Sphinx warnings-as-errors build.
- Wheel and source-distribution builds.

## Extension Compliance

- **Security Baseline**: Disabled; skipped.
- **Resiliency Baseline**: Disabled; skipped.
- **Property-Based Testing**: Partial; the removed Hypothesis strategy exercised
  only the deleted experimental benchmark result matrix. No production pure
  function or serialization contract was removed or added, so no replacement
  property test is required.
