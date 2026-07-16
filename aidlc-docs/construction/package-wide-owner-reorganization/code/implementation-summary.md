# Package-Wide Owner Reorganization Implementation Summary

The package now uses owner-oriented helper modules across domain classes,
synthesis schemas, service runtime records, graphing, and HEN equation/solver
internals. Public parents remain stable; explicitly selected runtime and
solver-state aliases were removed. A follow-up clean break removed the
synthesis compatibility modules and old barrel-qualified pickle paths.

Key changes include complete Stream, Value, ProblemTable, validation, and
workspace-view extractions; concrete synthesis schema ownership and typed lazy
barrels; private MVR/HPR/graph/dashboard/grid records and adapters; and explicit
HEN solver execution, evolution, pinch preprocessing, and extraction metadata
helpers.
