# Package Usability Refactor Persona

## Process Engineer

A process engineer applies pinch analysis, Total Site analysis, heat pumps,
refrigeration, MVR, energy transfer, exergy, cogeneration, area/cost targeting,
and HEN design to real process studies. Python experience may range from limited
to advanced, but package internals are never prerequisite knowledge.

The process engineer needs one consistent `PinchProblem` workflow that starts
with packaged or project input, uses stored engineering configuration when
method arguments are omitted, accepts explicit one-off overrides, and exposes
results, plots, designs, comparisons, and exports without private imports or
hidden analysis execution. The tutorials must support progression from a first
solve through every core and advanced public method.
