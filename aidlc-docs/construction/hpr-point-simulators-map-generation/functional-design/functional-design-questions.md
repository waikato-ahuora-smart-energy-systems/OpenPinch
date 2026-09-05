# Unit 2 HPR Point Simulators and Map Generation Questions

Please fill every `[Answer]:` tag with one listed letter. Choose `X` and add a
description when none of the listed options matches the intended model.

## Question 1
How should the default CoolProp simulator represent part-load operation at a
fixed source/sink temperature pair in the first release?

A) Preserve the current steady-state vapour-compression physics: useful duty
scales refrigerant mass flow, so COP is constant across load fractions at fixed
temperatures unless a future explicit degradation model is added. This is the
recommended compatibility-first behavior.

B) Apply a new built-in cycling/part-load degradation curve to CoolProp power,
even though current targeting does not define such a curve.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 2
What physical TESPy topology should define the first supported map generator?

A) A refrigerant-only closed loop with imposed evaporating/condensing
conditions and fixed approach-temperature assumptions; external source/sink
temperatures are translated to refrigerant conditions without explicit
secondary-fluid loops.

B) A single-stage refrigerant loop with explicit source-side and sink-side
secondary-fluid heat exchangers and circulation pumps, so external service
temperatures, heat-exchanger offdesign behavior, and modeled pump power are
inside the simulated boundary. This is the recommended physical boundary.

C) Defer the real TESPy leaf and implement only the protocol/fake adapter until
a project-specific TESPy network definition is supplied.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 3
How should TESPy establish the equipment design point before evaluating the
requested temperature/load grid?

A) Solve one global design point from the successful OpenPinch target's nominal
source temperature, sink temperature, and reference capacity, then evaluate
every requested coordinate as offdesign for that same equipment. This is the
recommended fixed-equipment interpretation.

B) Solve a separate full-load design point for every requested source/sink
temperature pair, then evaluate only lower loads as offdesign on that curve.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 4
Who should own the first-release TESPy characteristic curves and related
offdesign parameters?

A) OpenPinch should ship one explicit, versioned set of compressor-efficiency,
heat-exchanger, and pump characteristics and record their identifiers and
values in provenance. This is the recommended reproducible baseline.

B) Use TESPy's installed default characteristic data and record the TESPy
version and resolved characteristic identifiers in provenance.

C) Require callers to supply every characteristic explicitly; OpenPinch should
provide no default TESPy map model.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Question 5
What should `electric_power` include for the explicit-secondary-loop TESPy
model, if that topology is selected?

A) Compressor power plus every modeled electrical auxiliary, including
source/sink circulation pumps; provenance lists the included components. This
is the recommended total-electricity boundary for downstream balances.

B) Compressor power only; pump power remains outside the performance map and
provenance explicitly says auxiliaries are excluded.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: Not selected

## Question 6
When one or more requested operating points fail, how much work should the
all-or-nothing generator perform before raising an error?

A) Stop at the first deterministic grid failure, clean up the simulator, and
raise one point-specific error.

B) Attempt every requested point, collect ordered structured diagnostics, clean
up once, and raise one aggregate generation error without returning a partial
map. This is the recommended diagnostic-rich behavior.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: B
