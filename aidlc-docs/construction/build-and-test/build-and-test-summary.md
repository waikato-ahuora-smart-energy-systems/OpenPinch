# Build and Test Summary

## Build Status

- **Build tool**: Hatchling through the locked uv environment.
- **Build status**: Success.
- **Artifacts**: `dist/openpinch-0.5.4.tar.gz` and
  `dist/openpinch-0.5.4-py3-none-any.whl`.
- **Package smoke**: The fresh wheel was installed into an isolated import
  target and executed notebook 19 using the project's installed dependencies.

## Test Execution Summary

### Focused TDD Gates

- Public-signature RED failed because hierarchy routing and indirect placement
  scope did not yet exist; hierarchy GREEN passed the 12 focused routing and
  context tests.
- Existing-utility inference RED failed because no problem-to-template builder
  existed; inference GREEN passed classification, `Both` expansion,
  deterministic padding, explicit-count precedence, and declaration-order
  properties.
- Notebook and RTD RED failed on the former single-scope workflow; GREEN passed
  with separate Process/GCC and Site/TSP examples and no public target-mode
  selector.
- The generator byte-stability regression proves that all 19 checked-in
  notebooks are current without rewriting equivalent Python formatting or
  environment-owned kernel metadata.
- Ruff, `git diff --check`, and the warnings-as-errors Sphinx build: pass.

### Complete Repository Suite

- **Collected**: 2,408 tests.
- **Passed**: 2,404.
- **Skipped**: 4 expected environment/profile-specific tests.
- **Failed**: 0.
- **Solver coverage**: solver-marked tests were enabled with the configured
  local solver executables.
- **Duration**: 327.41 seconds.
- **Documentation**: the warnings-as-errors documentation build passed within
  the complete suite.

### Installed-Wheel Notebook

The installed wheel executed the sole utility-placement notebook from top to
bottom through the hierarchy-aware case API. The Process/GCC workflow selected
`Almond`, produced `0.06990104239676753 kW/K`, and returned eight optimized
utilities. The Site/TSP workflow defaulted to the Site master zone, produced
`1.1080402046277644 kW/K`, and returned eight optimized utilities. Both cases
registered through `workspace.add(...)`, preserved `baseline` as active, and
created standard Plotly figures. A separate wheel smoke inferred two
isothermal and two sensible levels per side from existing utilities without
count arguments. The imported `OpenPinch` path resolved to the isolated wheel
target.

## Delivered Scope

Utility placement is thermodynamic-only. Its public request, templates,
results, accessor, documentation, and notebook contain no
monetary selector, per-utility prices, cogeneration eligibility, electricity
price, turbine settings, or monetary breakdown. The physical balanced-
composite entropy objective, raw all-period weighted sum, default `HU`/`CU`
exclusion, named-case replacement, standard plots, and general OpenPinch
cogeneration analyses remain intact. No utility-placement CLI was added.
The final ownership audit removed generator-created cosmetic changes from
notebooks 1-18; only the required notebook 19 was added. RTD now describes the
thermodynamic-only case workflow and the exact 192-operation tutorial coverage
count. The public call accepts optional `isothermal`, `sensible`, and `zone`
arguments. Omitted `zone` uses the master zone; Process/Unit Operation routes
to direct GCC, Site routes to Total Site targeting, and Community/Region routes
to aggregate indirect targeting. Omitted counts infer typed templates from
existing utilities, while any explicit count selects generated-template mode.
The call returns a detached normal `PinchProblem`; the obsolete public
`base_target`, placement-specific presentation module, and observation methods
are absent.

## Overall Status

- **Build**: Success.
- **All tests**: Pass.
- **Enabled Property-Based Testing extension**: Compliant; analytical,
  generated-invariant, fixed-seed, round-trip, ordering, feasibility, and
  all-period properties remain green.
- **Security and Resiliency extensions**: Disabled; N/A.
- **Operations**: N/A; no deployment or publication was requested.
