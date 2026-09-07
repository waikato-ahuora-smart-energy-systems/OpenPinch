# HPR change audit

## Pass 1: integrity and canonical input

Confirmed and corrected three defect groups:

- HPR handle integrity omitted period index, utility HTC, fluid metadata and other
  copied physical quantities. Include selected-period quantities with units,
  period/zone identity and nominal segment fractions in the retained digest.
- Residual input accepted conflicting configured periods, shared-period HPR and
  ignored child/type/temperature-shift topology. Validate the single frozen basis
  and supply its period when options are omitted.
- Residual preparation lost canonical zone names and configured period weights;
  derivation omitted fluid, pressure and enthalpy metadata. Preserve these fields.

Evidence: six failing boundary cases reproduced before correction; affected HPR,
derived-case and new adversarial tests now pass (43 passed). Zero-load replay was
also checked: it correctly clears the current HPR result while allowing explicit
read-only access to a retained valid result. Source case inputs remain unchanged.

## Pass 2: cross-module interactions

Confirmed that partially specified, identity-aware capacity limits caused residual
case creation to fail in an uncapped period: the internal NaN sentinel leaked
into scalar canonical input. Convert that sentinel to an absent capacity; retain
finite limits. A real two-period HPR regression checks both periods and fluid,
pressure and enthalpy preservation. Also corrected “final allocationing” in
notebook 19 and its generator without changing saved outputs.

Validation: 113 tests pass, including all audit regressions, workspace persistence
and utility placement capacity tests. Ruff and patch hygiene pass.

## Pass 3: final complete review

Confirmed and fixed a combined period/unit transfer defect: converting an
implicit-unit singleton capacity vector collapsed it to a scalar but retained its
period IDs, making the derived input invalid. Preserve the original vector shape
and identity/weight metadata during unit binding. Regressions cover each partial
period selection and the complete reversed period vector (3 passed).

Both notebook 08/19 clean kernels, all 19 generator comparisons, strict Sphinx,
wheel/sdist and installed public API/artifact checks pass. The solver gate passes
3 tests with one optional skip. Rebuild and installed checks are repeated after
the final unit-binding correction. The initial complete-suite run was interrupted
to apply this correction; it is not final evidence. Its image-export failure was
a sandbox restriction launching Chrome. The fresh full coverage run permits the
required local browser and excludes only the two independently confirmed saved
notebook-output invariants. All other tests remain included.

Final full non-solver regression: 3152 passed, 3 skipped and
6 deselected in 610.11 seconds. The six deselections are four
solver-marked tests and only the two saved-output checks described below.
All 17 new audit regressions pass within this run. Fresh combined line/branch
coverage is 95.7786% (unchanged 95% gate). Lines:
28683/29524; branches:
8345/9136. Coverage is measured from the
final source, without merging measurements from the interrupted run.

Final wheel/sdist rebuilt successfully. The installed public HPR/allocation/
placement/transfer sequence and all three new unit/period capacity cases pass
outside the checkout. Every loaded OpenPinch module was verified to come from
the installed wheel. Ruff, all 46 changed Python file formats and diff hygiene
pass. Notebook outputs and unrelated edits were preserved.

Stopping condition: three audit-and-repair passes completed as requested. All
identified implementation defects are fixed and verified; a fourth pass was not
requested. Two existing notebook-output checks still fail on notebook 06 because
they require cleared execution counts/outputs. Their assertions were not weakened
and saved evidence was not deleted. No commits, pushes or deployment performed.

Evidence: /tmp/hpr-audit-full-final.log, /tmp/hpr-audit-coverage.json,
/tmp/hpr-audit-coverage.log, /tmp/hpr-audit-kernels.log,
/tmp/hpr-audit-installed-final.log, /tmp/hpr-audit-solver.log,
/tmp/hpr-audit-saved-outputs.log. Final distributions:
/tmp/openpinch-audit-final-dist; installed copy: /tmp/openpinch-audit-installed.


No diagrams or embedded transport payloads in this report; Markdown inspected.
PBT enabled; existing state-sequence and unit/period transformation properties run
with the real workflow regressions. Disabled extensions remain skipped.
