# Indirect Profile Precision Implementation Summary

## Outcome

Direct Integration graph generation now rounds deep-copied Problem Tables, so
the runtime target retains its full-precision shifted and real cascades. The
indirect profile reconstruction path consumes the exact child `pt` temperature
and `H(net)-actual` columns without rounding reconstructed stream temperatures.

## Modified Code

- `OpenPinch/analysis/targeting/direct.py`: graph serialization rounds copies
  instead of mutating analysis tables.
- `OpenPinch/analysis/targeting/indirect.py`: reconstructed segment temperatures
  retain the cascade precision used to calculate indirect targets.
- Direct and indirect targeting tests cover non-mutating graph serialization,
  graph rounding, close temperature intervals, and five-decimal duties.

## Compatibility

No public API, schema, graph name, or rounding presentation changed. The two
per-Zone net-profile pairs and no-child fallback remain intact. Existing
coarse-resolution targeting results are preserved; fine-resolution inputs no
longer lose cascade topology or duty precision.

## Focused Evidence

The regression-first tests failed twice before implementation and now pass.
The expanded direct, indirect, hierarchy, graph, service, multi-scale,
multi-period, HPR, and model round-trip matrix passes 158 tests with fixed
Hypothesis seed 20260715. Focused Ruff lint and formatting pass.
