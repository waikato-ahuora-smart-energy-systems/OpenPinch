# Heat-Pump Zero-Duty CI Code Generation Plan

## Plan Authority

This checklist is the single source of truth for Code Generation of the GitHub
CI zero-duty heat-pump regression. Execute steps in order and mark every
checkbox complete in the same interaction as its work.

## Unit Context

- **Unit**: Vapour-compression process-stream emission.
- **Workspace**: `/Users/timothyw/Github_Local/OpenPinch`.
- **Project type**: Brownfield Python library.
- **Approved requirements**:
  `aidlc-docs/inception/requirements/github-ci-heat-pump-zero-duty-requirements.md`.
- **Approved workflow**:
  `aidlc-docs/inception/plans/github-ci-heat-pump-zero-duty-execution-plan.md`.
- **User stories**: N/A; explicitly skipped for this isolated internal bug fix.
- **Primary dependency**: CoolProp-derived temperature-enthalpy profiles.
- **Preserved contract**: Generic segmented streams require strictly monotonic,
  positive-duty heat coordinates.
- **Owned data entities**: None.
- **Public interfaces**: No changes.
- **Database, API, frontend, deployment, and documentation surfaces**: N/A.

## Exact File Scope

- Modify:
  `OpenPinch/services/heat_pump_integration/unit_models/vapour_compression_cycle.py`.
- Modify:
  `tests/test_classes/test_simple_heat_pump_cycle.py`.
- Create:
  `tests/strategies/heat_pump_cycles.py`.
- Create:
  `aidlc-docs/construction/github-ci-heat-pump-zero-duty/code/code-generation-summary.md`.
- Do not modify generic profile validation, cascade production code,
  dependencies, GitHub workflows, or unrelated dirty HEN benchmark files.

## Requirements Traceability

- **FR-CI-01 / FR-CI-02**: Steps 1 through 3 cover condenser and evaporator
  zero-duty omission.
- **FR-CI-03**: Steps 2 and 3 preserve non-zero and negative process duties.
- **FR-CI-04**: Step 5 verifies the existing cascade delegation tests.
- **FR-CI-05**: Step 3 leaves the strict generic validator unchanged; Step 5
  verifies its existing tests.
- **NFR-CI-01 / NFR-CI-04**: Steps 1 and 2 inject deterministic one-ULP
  profiles independent of CoolProp platform behavior.
- **NFR-CI-03**: Step 5 exercises existing positive, negative, and cascade
  cases.
- **NFR-CI-05**: Steps 1 and 2 add constrained Hypothesis invariant coverage.

## Planning Status

- [x] Read approved requirements and workflow artifacts.
- [x] Read brownfield code structure and confirm workspace code locations.
- [x] Inspect the target unit model, reported tests, existing strategy package,
  and shared profile invariant.
- [x] Confirm target production and test files have no pre-existing local diff.
- [x] Identify dependencies, preserved contracts, and non-applicable layers.
- [x] Prepare and content-validate this executable checklist.
- [x] Obtain explicit approval for the complete Code Generation plan.

## Generation Steps

### Step 1 — Add a domain-specific Hypothesis strategy

- [x] Create `tests/strategies/heat_pump_cycles.py`.
- [x] Define an immutable zero-duty stream-side case containing the side,
  finite duty within `[-tol, tol]`, positive finite mass flow, and a finite
  one-ULP temperature-enthalpy profile.
- [x] Implement a composite Hypothesis strategy with constrained,
  thermodynamically plausible ranges and both condenser and evaporator sides.
- [x] Leave shrinking enabled and rely on the repository's fixed CI seed.

### Step 2 — Add deterministic and invariant regression tests

- [x] Update `tests/test_classes/test_simple_heat_pump_cycle.py` in place.
- [x] Add a deterministic parameterized regression for condenser and
  evaporator profiles whose enthalpy coordinates differ by one
  `numpy.nextafter` step while the corresponding external duty is zero.
- [x] Assert the resulting collection is empty and that the property-profile
  builder is not invoked for a zero external process duty.
- [x] Add a Hypothesis property test using the domain strategy: every generated
  process side with `abs(duty) <= tol` emits no stream.
- [x] Retain existing examples as coverage that positive and intentional
  negative duties still emit correctly classified streams.

### Step 3 — Implement the narrow production repair

- [x] Import the shared project `tol` constant in
  `vapour_compression_cycle.py`.
- [x] Before building a condenser profile, require
  `abs(float(self._Q_heat)) > tol`.
- [x] Before building an evaporator profile, require
  `abs(float(self._Q_cool)) > tol`.
- [x] Retain the existing positive-mass-flow and exact profile-span guards as
  defense in depth.
- [x] Do not relax, bypass, or modify shared segmented-profile monotonicity
  validation.
- [x] Review the diff to confirm negative duties are preserved by absolute
  magnitude and no public interface changes were introduced.

### Step 4 — Record the generated-code summary

- [x] Create
  `aidlc-docs/construction/github-ci-heat-pump-zero-duty/code/code-generation-summary.md`.
- [x] List modified and created files, requirements traceability, the numerical
  rationale, and the tests generated.
- [x] State explicitly that test execution belongs to the subsequent Build and
  Test stage.

### Step 5 — Close Code Generation

- [x] Verify no duplicate or alternate production files were created.
- [x] Run whitespace and syntax-oriented diff checks on the scoped files.
- [x] Confirm unrelated working-tree changes remain untouched.
- [x] Mark all completed generation checkboxes and the Code Generation state in
  `aidlc-docs/aidlc-state.md`.
- [x] Log completion and present the standardized Code Generation review gate.

## Subsequent Build and Test Scope

After Code Generation approval, Build and Test will run:

1. The three exact GitHub Actions failures.
2. Deterministic and Hypothesis zero-duty regression tests.
3. The full simple and cascade heat-pump cycle modules.
4. Existing stream-profile invariant tests.
5. Ruff format/lint checks for scoped Python files.
6. The full supported non-solver pytest suite with the CI Hypothesis seed.

## Property-Based Testing Compliance

- **PBT-01 (advisory in Partial mode)**: Compliant; the business invariant is
  explicitly identified.
- **PBT-02**: N/A; no logical inverse or round-trip operation is changed.
- **PBT-03**: Compliant in plan; Step 2 tests zero-duty omission across
  generated domain cases.
- **PBT-04 (advisory)**: N/A; no idempotent operation is claimed.
- **PBT-05 (advisory)**: N/A; there is no independent oracle algorithm.
- **PBT-06 (advisory)**: N/A; no command-sequence state machine is changed.
- **PBT-07**: Compliant in plan; Step 1 defines constrained domain cases rather
  than unconstrained primitive generators.
- **PBT-08**: Compliant in plan; shrinking remains enabled and CI uses
  `--hypothesis-seed=20260715`.
- **PBT-09**: Compliant; Hypothesis is selected, locked, and integrated with
  pytest.
- **PBT-10 (advisory)**: Compliant in plan; the concrete one-ULP example
  complements the generated invariant test.

No blocking PBT finding exists at the planning gate.

## Other Extension Compliance

- **Security Baseline**: Disabled; skipped.
- **Resiliency Baseline**: Disabled; skipped.
