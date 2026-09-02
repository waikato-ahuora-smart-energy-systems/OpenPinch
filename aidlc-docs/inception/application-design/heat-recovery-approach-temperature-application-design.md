# Heat-Recovery Approach Temperature Application Design

## Component boundaries

- `OpenPinch.contracts.heat_recovery` owns the specialist immutable result
  contract and status vocabulary.
- `OpenPinch.analysis.targeting.approach_temperature` owns detached cascade
  evaluation, bracketing, tolerances, and deterministic bisection.
- `OpenPinch.application.heat_recovery_approach` owns zone and period
  resolution, input/output units, all-period concurrency, and public errors.
- Existing target accessors provide thin problem, workspace, and case-batch
  delegation without adding package-root exports.

## Data flow

The application service resolves the requested zone and canonical period,
normalizes the recovery request to canonical heat-flow units, and sends deep
copies of process streams to the numerical solver. The solver evaluates the
zero-approach limit and no-overlap bound, bisects the feasible interval, and
returns canonical numerical values. The application layer converts values to
configured output units and constructs the public frozen result.

## Business rules

- Site, Process Zone, and Unit Operation scopes are supported.
- Community and Region scopes are rejected with direct-targeting guidance.
- All-period mappings must contain exactly the canonical period IDs.
- Each period and batch case is isolated; ordering follows canonical input
  order.
- No operation writes to problem-owned target or result state.
