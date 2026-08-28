# Utility Placement Optimisation Requirements Questions

Please answer every question by entering one letter after its `[Answer]:` tag.
When choosing `X`, add the custom response on the same line or immediately
below it.

## Question 1
How should the requested number of utility levels be specified?

A) One count for hot utilities and one count for cold utilities, with each count having a minimum of two (recommended)

B) One shared count applied independently to both the hot and cold utility sides

C) One total count that the optimiser divides between hot and cold utilities

X) Other (please describe after the [Answer]: tag below)

[Answer]: B

## Question 2
What physical utility model should the first release optimise?

A) Isothermal utility levels represented by one temperature each, suitable for condensing steam and evaporating refrigerants (recommended)

B) Sensible utility levels with separate supply and target temperatures

C) Both isothermal and sensible utilities selected explicitly by the caller

X) Other (please describe after the [Answer]: tag below)

[Answer]:  B

## Question 3
Which heat-integration scope should utility placement support in the first release?

A) Direct integration for one selectable zone and period (recommended first increment)

B) Indirect or Total Site integration only

C) Both direct and indirect or Total Site integration through one service

X) Other (please describe after the [Answer]: tag below)

[Answer]: C

## Question 4
Which variables should the optimiser choose?

A) Utility temperatures only; the existing cascade determines feasible duties at each candidate level (recommended)

B) Utility temperatures and duties jointly

C) Utility temperatures, duties, and utility technology or fluid identity jointly

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 5
How should the temperature search bounds be defined?

A) Derive feasible bounds from shifted process or site profiles and minimum approach temperatures, with optional caller overrides (recommended)

B) Require the caller to supply explicit lower and upper bounds for every level

C) Use configuration defaults independent of the solved process profile

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 6
What ordering and separation constraint should apply between adjacent utility levels?

A) Enforce strict hot-descending and cold-ascending order with a configurable positive minimum separation (recommended)

B) Enforce ordering only and allow coincident utility levels

C) Do not constrain ordering during optimisation; sort and merge the result afterward

X) Other (please describe after the [Answer]: tag below)

[Answer]:  A

## Question 7
What should the default thermodynamic cost represent?

A) Total entropy generation from utility-to-process heat transfer, with exergy destruction also reported as ambient temperature multiplied by entropy generation (recommended)

B) Exergy destruction or lost work as the primary objective, with entropy generation reported secondarily

C) Net external exergy requirement after crediting cogenerated power

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 8
Which objective modes should the public service expose?

A) Two scalar modes: thermodynamic and monetary, with thermodynamic as the default (recommended)

B) Thermodynamic, monetary, and a caller-weighted hybrid objective

C) Thermodynamic and monetary objectives plus a Pareto frontier rather than a weighted hybrid

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 9
What should the monetary objective include in the first release?

A) Purchased thermal utility cost minus cogenerated electricity credit, excluding new equipment capital cost (recommended)

B) Thermal utility cost, electricity import or export, and annualised turbine or utility-system capital cost

C) Thermal utility operating cost only, with cogeneration reported but not credited in the objective

X) Other (please describe after the [Answer]: tag below)

[Answer]:  A

## Question 10
How should cogeneration be modelled for the monetary objective?

A) Reuse OpenPinch's existing configurable multi-stage steam-turbine cogeneration service (recommended)

B) Use a new ideal isentropic or Carnot power-credit calculation specific to utility placement

C) Defer cogeneration and implement monetary placement from thermal utility prices only

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 11
How should monetary prices and operating assumptions enter the service?

A) Explicit method arguments override existing configuration defaults; utility prices may also be supplied per candidate utility family (recommended)

B) Require every monetary input as an explicit method argument with no configuration fallback

C) Read all monetary assumptions from the existing problem configuration only

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 12
What result detail should the service return?

A) A typed best solution plus ordered alternative candidates, each with temperatures, duties, feasibility, entropy or exergy terms, monetary components, cogeneration, and objective value (recommended)

B) Only the best utility temperatures, duties, and scalar objective value

C) A full optimisation trace containing every evaluated candidate in addition to the best solution

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 13
Should the optimised utilities mutate the problem's existing utility collection?

A) No; return a detached result and require an explicit later action to apply it (recommended)

B) Yes; replace the selected zone's utilities automatically after a successful solve

C) Allow the caller to choose with an `apply` argument that defaults to false

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 14
What operating-period behavior is required initially?

A) Optimise one selected period; multiperiod placement is deferred (recommended first increment)

B) Choose one shared set of utility temperatures across all periods using weighted aggregate cost

C) Optimise an independent utility placement for every period

X) Other (please describe after the [Answer]: tag below)

[Answer]: B

## Question 15
How should infeasible candidates and a fully infeasible optimisation be handled?

A) Reject or penalise individual infeasible candidates, and raise a typed actionable error if no feasible solution exists (recommended)

B) Return a typed unsuccessful result with diagnostics and never raise for mathematical infeasibility

C) Return the least-infeasible candidate with quantified constraint violations

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 16
What public workflow shape should be targeted?

A) `problem.target.utility_placement(...)`, mirrored by workspace case batches and specialist concrete imports (recommended)

B) A specialist service imported only from `OpenPinch.analysis`, with no application accessor initially

C) A new top-level root API object dedicated to utility placement

X) Other (please describe after the [Answer]: tag below)

[Answer]:  A

## Question 17
What optimisation execution policy should be the default?

A) Deterministic fixed-seed bounded optimisation using the existing solver-neutral service, with tolerances and backend overrides exposed (recommended)

B) A deterministic exhaustive or structured grid search only

C) Automatically compare every installed optimisation backend and choose the best result

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 18
What evidence should anchor the TDD acceptance suite?

A) Hand-calculable analytical fixtures plus regression cases derived from existing OpenPinch sample problems (recommended)

B) Published utility-placement literature examples as the primary numerical oracle

C) Differential comparison against an existing external utility-placement implementation

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 19
Should the resiliency baseline be applied to this project?

**What this extension is.** Enabling it applies directional, design-time best
practices for fault tolerance, availability, observability, recoverability, and
continuous improvement. It is a starting point, not production certification.

A) Yes — apply the resiliency baseline as directional design-time guidance

B) No — skip the resiliency baseline (recommended for this in-process numerical library feature)

X) Other (please describe after the [Answer]: tag below)

[Answer]:  B

## Question 20
Should property-based testing rules be enforced for this project?

A) Yes — enforce all property-based testing rules as blocking constraints (recommended for this numerical optimisation feature)

B) Partial — enforce them only for pure functions and serialization round-trips

C) No — skip property-based testing rules

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 21
Should security extension rules be enforced for this project?

A) Yes — enforce all security rules as blocking constraints

B) No — skip the security baseline (recommended for a local in-process numerical feature with no new I/O or network boundary)

X) Other (please describe after the [Answer]: tag below)

[Answer]: B
