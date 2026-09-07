# Unit and property gates

Run `.venv/bin/python -m pytest -q tests/application/test_hpr_change_audit.py tests/application/test_hpr_derived_case_api.py tests/application/test_hpr_residual_workflow.py tests/application/test_analysis_contracts.py tests/application/test_analysis_reliability.py --hypothesis-seed=20260715`. Also run utility-placement, package API and documentation contract tests. Existing pytest/Hypothesis shrinking is enabled.


The repeated-audit regressions cover modified target metadata, frozen period and
zone validation, default-period reconstruction, retained names/weights, zero-load
replay, fluid/pressure/enthalpy transfer, uncapped periods and singleton capacity
arrays under unit conversion and period reordering.

For the complete gate, set `OPENPINCH_TUTORIAL_PROFILES=all` and run coverage
with branch measurement over OpenPinch and pytest with Hypothesis seed 20260715.
Do not exclude test names or markers. Saved notebook artifacts are validated
against the Jupyter schema, while fresh generation must have empty execution
state. Permit local Chrome execution for the existing image-export test.
See `../all-tests-repair-summary.md` for the complete-suite follow-up.
