# Superseded Heat-Recovery Notebook Approach Correction Summary

This records a provisional 12,000 kW notebook-only workaround. The user
corrected its thermodynamic premise before commit: a threshold problem can
retain maximum recovery at a positive global `dt_min`. The authoritative
solver and notebook outcome is recorded by the threshold-limit correction.

## Root cause and correction

The Bleaching ordinary direct target is 14,121.972 kW, equal to the inverse
service's zero-`dt_min` thermodynamic limit. The provisional implementation
incorrectly assumed that this required a zero inverse result.

The temporary notebook used a 12,000 kW interior requirement. It has since been
replaced by direct inversion of the maximum-recovery target, which returns the
greatest feasible threshold approach.

## TDD and verification

- A failing executable notebook regression reproduced the 14,121.972 kW
  boundary request before the correction.
- The regression now pins the 12,000 kW request, thermodynamic limit,
  61.8673468 ``delta_degC`` result, and ``solved`` status.
- The focused notebook and documentation gate passes 47 tests with 3 expected
  optional-profile skips, including notebook execution, generator drift, and
  warning-strict Sphinx.
- Ruff lint, formatting, and Git patch hygiene pass.

The user's executed notebook that exposed the issue remains preserved at
``/tmp/OpenPinch-02-focused-direct-and-total-site-executed-20260902.ipynb``.
