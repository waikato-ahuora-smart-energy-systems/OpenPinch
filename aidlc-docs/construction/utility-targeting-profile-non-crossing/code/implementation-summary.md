# Utility Targeting Profile Non-Crossing Implementation Summary

## Outcome

Shared Process utility targeting now enforces the tightest current-end
temperature limit whenever a sensible utility target lies within the GCC
range. Ordinary direct targeting and utility-placement exact replay therefore
cannot return a sensible Utility GCC that crosses the residual Process GCC at
an interior breakpoint.

## Root cause and correction

`_maximise_utility_duty` calculated adjacent- and current-end profile limits,
but its selection condition inspected the adjacent-end temperature limit. It
could consequently discard a tighter current-end limit and allocate the total
endpoint duty. The correction uses the finite current-end temperature limit
while retaining the established endpoint result for utilities located wholly
outside the GCC range. This preserves high-temperature isothermal fallback and
maximum-duty behavior.

The notebook-derived counterexample previously allocated 139.2164 kW to one
hot sensible level. Corrected ordinary targeting allocates approximately
125.6384 kW to that sensible level and 13.5780 kW to the adjacent isothermal
level. At 97 degC the corrected utility profile and Process GCC both equal
approximately 38.5152 kW; the maximum observed excess is numerical noise.

## TDD evidence

- RED: the notebook-derived example and a shrunk hot/cold Hypothesis example
  both failed before the production correction.
- GREEN: 67 shared utility/direct-targeting tests pass.
- Analytical and specialist placement gate: 248 tests pass.
- Application exact-replay selection: 7 tests pass, including Process GCC
  non-crossing, ordinary-retarget equality, multiperiod replay, and caps.
- Notebook and documentation gate: 20 tests pass.
- Complete solver-enabled repository gate: 2,450 passed and 4 expected skips
  in 526.67 seconds.

## Delivered files

- Corrected shared targeting in `OpenPinch/analysis/targeting/utilities.py`.
- Added example, property, and application exact-replay regressions.
- Updated requirements, RTD, notebook generator, and assertion-free executable
  notebook 19 with the standard GCC and Total Site Profile workflows.
- No public API, monetary optimization, CLI, dependency, infrastructure, or
  placement-specific plotting surface was added.

## Extension compliance

- PBT-03: compliant; generated hot and cold sensible profiles remain within
  their GCC envelopes at every generated breakpoint.
- PBT-05: compliant; the notebook-derived piecewise-linear duty limit is pinned
  to its analytical minimum-ratio value.
- PBT-07 and PBT-08: compliant; bounded domain inputs use normal Hypothesis
  shrinking and repository seed reporting.
- PBT-10: compliant; explicit regression and generated invariant both exist.
- PBT-02, PBT-04, and PBT-06: N/A; no inverse, idempotent, or stateful operation
  changed. PBT-09 remains satisfied by the existing Hypothesis stack.
- Security and Resiliency: disabled and N/A.
