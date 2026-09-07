# HPR accounting code generation

Authorized through completion. Stories HPR-US-1, 2, 4; R1/R3/R5.
Existing Python library; no database, deployment or UI artifact applies.

- [x] 1. Define numerical rules and immutable records in functional design.
- [x] 2. Add focused regressions for residual grid, fraction bounds and ambient load ceiling.
- [x] 3. Correct OpenPinch/analysis/heat_pumps common postprocessing, service and Carnot targeting; add OpenPinch/domain/hpr.py records and domain/targets.py fields.
- [x] 4. Verify focused HPR tests and complementary Hypothesis oracle/invariant properties with seed 20260715; fix demonstrated regressions.
- [x] 5. Record numerical evidence and handoff contracts in code summary; update state.

PBT properties: finite aligned profiles, selected duty bounds, energy balance,
no unused-cold HP penalty, immutable typed-record transport. Use independent
arithmetic/cascade cases, not stochastic optimizer exact outputs. No new stateful
component in this unit; state properties belong to unit 3. Hypothesis shrinking
and existing CI configuration retained. Security/Resiliency disabled.
