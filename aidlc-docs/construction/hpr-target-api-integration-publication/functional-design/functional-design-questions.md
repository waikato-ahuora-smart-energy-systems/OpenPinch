# Unit 3 Functional Design Questions

Please answer the following question to resolve the remaining backend-selection
ambiguity before the Unit 3 functional artifacts are generated.

## Question 1
When `simulation_backend="tespy"` is passed to an ordinary current
`vapour_compression_heat_pump` or `vapour_compression_refrigeration` targeting
call, what must TESPy change before the target is returned?

A) The ordinary target calculation remains the established CoolProp-backed HPR
optimization; the selector records which simulator the later explicit
`hpr_performance_map` call must use. TESPy runs only during that follow-up map
generation.

B) TESPy must replace the thermodynamic cycle evaluations inside the ordinary
HPR targeting optimization, so target selection and returned target numbers are
TESPy-backed. This expands the approved work beyond Unit 2's existing
target-derived point-simulator interface.

C) The ordinary optimization remains CoolProp-backed, but TESPy must re-simulate
and validate the selected nominal target before it is returned. A failed or
inconsistent TESPy nominal solve rejects the target; the later map call also
uses TESPy.

X) Other (please describe after [Answer]: tag below)

[Answer]: B

## Why This Decision Is Required

The approved design uses `simulation_backend` on the current targeting methods,
while completed Unit 2 intentionally owns point and map simulation rather than
the HPR targeting optimizer. Option A preserves the completed unit boundary but
makes the selector map-simulation intent. Option B makes it a full targeting
engine selector and requires a broader adapter design. Option C adds a nominal
TESPy acceptance gate without replacing target optimization.

## Content Validation

This question file contains no Mermaid diagram, ASCII diagram, executable code
block, JSON, or YAML. Markdown headings, choices, blank-line separation,
special characters, and the answer tag were checked before creation.
