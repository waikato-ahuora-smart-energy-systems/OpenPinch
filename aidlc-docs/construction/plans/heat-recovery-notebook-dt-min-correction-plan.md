# Superseded Heat-Recovery Notebook Approach Correction Plan

This provisional notebook-only workaround was completed locally but superseded
before commit by the user's threshold-problem correction. The authoritative
implementation is
``heat-recovery-threshold-limit-correction-plan.md``; the 12,000 kW workaround
is not part of the final notebook or service contract.

## Diagnosis

The provisional diagnosis treated every thermodynamic-limit inverse as zero
approach. That assumption was incomplete because threshold problems can retain
maximum recovery over a positive global-approach plateau.

## TDD execution

- [x] Add a failing executable regression that requires the notebook example
  to solve the explicit 12,000 kW interior request at approximately
  61.86735 ``delta_degC`` with status ``solved``.
- [x] Correct the canonical notebook generator and teaching text to distinguish
  the ordinary target, thermodynamic-limit boundary, and interior request.
- [x] Regenerate the output-free committed notebook while preserving the
  user's executed copy in ``/tmp``.
- [x] Run the focused regression, notebook execution and drift suites, Ruff,
  warning-strict Sphinx, and patch-hygiene checks.
- [x] Record completion evidence and update workflow state.

## Extension compliance

Property-Based Testing is enabled. Existing solver properties continue to
cover monotonicity and boundary behavior; no new algorithmic behavior is
introduced, so PBT-01 through PBT-10 are N/A for this notebook-only correction.
Security and Resiliency remain disabled.
