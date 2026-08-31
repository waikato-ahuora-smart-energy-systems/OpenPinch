# Utility Placement Requirements Approval

The updated comprehensive requirements define separate isothermal and sensible
level counts, complete per-period heating and cooling coverage, utility
templates, decision variables, thermodynamic and monetary objectives,
multiperiod semantics, public results, and TDD/PBT acceptance evidence.

## Question 1
How should the workflow proceed from this requirements checkpoint?

A) Approve the requirements and continue to User Stories

B) Request changes to the requirements before continuing

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Profile-Gap Correctness Amendment

The user's continuing authorization approves acceptance criterion 18 and the
corrected FR-009 interpretation: default thermodynamic placement minimizes the
entropy-weighted horizontal separation between the residual process and
allocated utility profiles. Whole-process entropy independent of the candidate
placement is excluded from the objective.

## Balanced-Composite Entropy Amendment

The user's continuing authorization supersedes the reciprocal-profile
surrogate. Default thermodynamic placement now minimizes physical entropy
generation from candidate-local real-temperature balanced composite curves,
using `CP * ln(T_out / T_in)` for sensible intervals and signed `Q / T` for
isothermal intervals. Positive allocation to generated `HU` or `CU` fallback
utilities receives a deterministic infeasible penalty.

## Thermodynamic-Only Scope Amendment

The user's continuing authorization supersedes the paused boiler/turbine
monetary amendment and defers monetary utility-placement optimisation in full.
The current public service has one objective: period-weighted physical entropy
generation from balanced composite curves. Per-utility price fields,
cogeneration eligibility, electricity-price and turbine inputs, monetary result
fields, and a monetary notebook workflow are excluded for now. General OpenPinch
cogeneration analyses outside utility placement are unchanged.

## API Usability Amendment

The user's continuing authorization approves the clarified public workflow.
`problem.target.utility_placement(isothermal=2, sensible=2)` returns a detached
normal `PinchProblem` containing the best optimized utilities and retains
detailed evidence at `optimized_case.utility_placement_result`. The source is
unchanged. `workspace.add(optimized_case, name="optimized_utilities",
activate=False)` explicitly registers it. Normal targeting, summaries,
reports, and standard plotting remain explicit case operations. The verbose
`*_level_count` arguments, detailed-result return, placement-specific
observation methods, nested-result notebook traversal, and manual utility
dictionary construction are superseded.

## Hierarchy and Existing-Utility Inference Amendment

The user's continuing authorization approves the final hierarchy contract.
`problem.target.utility_placement()` defaults to the problem master zone and
accepts an optional zone name, address, or object. The selected zone type
determines direct GCC, Site Total Site Profile, or Community/Region aggregate
indirect targeting, so `base_target` is removed from the public workflow.
Omitted counts infer typed templates from existing utilities; supplied counts
replace them with generic paired hot/cold levels. Ordinary thermal span defines
isothermal versus sensible behavior, `Both` expands to opposite-side pairs, and
asymmetric inferred inventories are padded deterministically. The notebook
demonstrates separate Process/GCC and Site/TSP placements.
