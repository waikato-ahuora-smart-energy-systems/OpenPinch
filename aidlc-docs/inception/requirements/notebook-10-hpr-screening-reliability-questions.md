# Notebook 10 HPR Screening Reliability Questions

Please complete every `[Answer]:` tag. The recommended choices reflect the
reproduced failure and the repository's source-only notebook contract.

## Question 1
What should define the Notebook 10 correction?

### Decision context

Notebook 10 asks whether one heat-pump concept remains useful across all
operating periods. There are two legitimate teaching goals:

- **Successful workflow demonstration**: users should see how to execute and
  inspect each public multiperiod HPR method without mistaking an avoidable
  topology mismatch for a software failure.
- **Feasibility screening**: users should also learn that a more complex HPR
  concept may be physically infeasible for one or more periods, and that this
  is an engineering result rather than an exception to hide.

The reproduced failures do not establish that vapour compression or MVR is
unsuitable for the example. All three methods solve across every period when
the notebook explicitly requests one condenser, one evaporator, one MVR stage,
20 iterations, and 50 evaluations. The current failures arise because the
notebook silently inherits a three-condenser/two-evaporator cascade and very
large default search limits.

Independent `target.all_periods` replay permits a separately optimized design
for each period. That answers whether a technology can be retuned for each
operating condition, but not whether one installed design can serve the whole
profile. Enabling `HPR_MULTIPERIOD_OPTIMIZATION_ENABLED` instead evaluates one
shared design vector in every period. Its objective combines weighted operating
cost and feasibility penalty with peak annualized capital cost. The returned
target exposes the shared vector, ordered period outputs, period weights, and
weighted result through `hpr_details`.

The bounded shared-design probe solved Carnot heat pumping, Carnot
refrigeration, vapour-compression heat pumping, vapour-compression
refrigeration, and VC+MVR across `turndown`, `base`, and `peak`.

A) Make bounded shared-design multiperiod optimization the main workflow for
all five technologies, then add one clearly labelled optional advanced cascade
screen that may report structured infeasibility evidence (recommended)

B) Make bounded shared-design multiperiod optimization the complete tutorial;
explain infeasibility in prose but do not execute an intentionally difficult
advanced screen

C) Retain independent `target.all_periods` replay as the main workflow, using
explicit single-stage settings for each separately optimized period

D) Apply the smallest patch: change only the three currently failing
independent calls to the verified single-stage settings, without restructuring
the tutorial around a shared design

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 2
How should the checked-in notebook's current execution counts and saved outputs
be handled?

A) Regenerate Notebook 10 as the canonical source-only artifact, clearing the
saved execution counts and outputs after the corrected workflow is verified
(recommended and required by the existing packaging test)

B) Preserve the current executed output in a separate diagnostic artifact, then
regenerate Notebook 10 as source-only

C) Retain the executed outputs inside the checked-in notebook even though this
conflicts with the repository's source-only notebook contract

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 3
What failure information should `screen_periods` retain if a future HPR screen
is infeasible?

A) Return an accurate failure status, the error message, and the bounded
structured `HPRTargetingError.diagnostics` as plain JSON (recommended)

B) Keep only the current generic status and string reason

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 4
Should security extension rules be enforced for this correction?

A) Yes - enforce all Security Baseline rules as blocking constraints

B) No - retain the project's existing disabled Security Baseline setting
(recommended for this local tutorial correction)

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: B

## Question 5
Should Property-Based Testing rules be enforced for this correction?

A) Yes - retain the project's existing full PBT setting and apply relevant
rules as blocking constraints (recommended)

B) Partial - enforce PBT only for pure functions and serialization round trips

C) No - skip PBT rules for this correction

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 6
Should the Resiliency Baseline be applied to this correction?

A) Yes - apply the baseline as directional design-time guidance

B) No - retain the project's existing disabled Resiliency Baseline setting
(recommended for this local tutorial correction)

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: B
