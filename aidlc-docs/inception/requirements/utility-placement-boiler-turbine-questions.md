# Utility Placement Boiler-Turbine Monetary Objective Questions

The monetary objective is being corrected so users do not price individual
utility levels. Please answer each question by placing a letter after its
`[Answer]:` tag. Property-Based Testing remains enabled; Security and
Resiliency remain disabled according to the existing project configuration.

## Question 1
How should boiler fuel demand be calculated for each period?

A) Use a shared-system first-law balance: boiler steam duty equals total hot utility heat delivered plus turbine shaft work, and fuel input equals boiler steam duty divided by boiler efficiency (recommended)

B) Calculate steam mass flow in the turbine and use an explicit feedwater state to obtain the boiler enthalpy rise before dividing by boiler efficiency

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 2
Which optimized hot utility levels belong to the shared boiler-turbine system?

A) Every positive-duty hot utility level is a turbine extraction from the shared boiler steam header; remove per-level cogeneration eligibility (recommended)

B) Allow some hot levels to bypass the turbine using a non-cost eligibility flag

X) Other (please describe after the [Answer]: tag below)

[Answer]:  A

## Question 3
Which economic and boiler inputs should monetary placement require?

A) Require one fuel price in currency per MWh of fuel, one boiler efficiency, and one electricity export price; keep the existing turbine inlet and performance settings (recommended)

B) Require fuel price and electricity price but read boiler efficiency from existing project configuration

X) Other (please describe after the [Answer]: tag below)

[Answer]:  A

## Question 4
How should turbine shaft work become credited electricity?

A) Add one generator efficiency and calculate credited electricity as turbine shaft work multiplied by generator efficiency (recommended)

B) Treat existing turbine work as net electrical power and credit it directly

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 5
How should cold utility operating cost be treated in this correction?

A) Exclude cold-side operating cost; optimize the boiler fuel cost minus generated-electricity credit while retaining cold levels only for heat-balance feasibility (recommended)

B) Add one centralized cooling-system electricity coefficient in kW electric per kW cooling

C) Add a separate centralized cooling-system model with its own efficiency and energy price

X) Other (please describe after the [Answer]: tag below)

[Answer]:  This raise the question whether the full utility system structure should be specified.

## Question 6
How should the public monetary contract change?

A) Remove per-template `utility_price` and `cogeneration_eligible` inputs from utility placement; replace thermal purchase cost with boiler duty, fuel input, fuel cost, turbine work, generated electricity, electricity credit, and net operating cost (recommended)

B) Stop using per-template prices but retain the old fields as deprecated, ignored compatibility inputs and add the new boiler-system breakdown

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 7
Should boiler/turbine capacity limits or capital costs enter this objective?

A) No; retain an operating-cost-only objective with unconstrained equipment capacity in this release (recommended)

B) Add maximum boiler fuel input and maximum turbine inlet-flow constraints, but exclude capital cost

C) Add equipment capacity constraints and annualized capital cost

X) Other (please describe after the [Answer]: tag below)

[Answer]: A
