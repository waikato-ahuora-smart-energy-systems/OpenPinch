# Business Rules

1. The supported workflow starts from `from OpenPinch import PinchProblem,
   PinchWorkspace`.
2. Core analysis runs only through descriptive `target` or `design` methods.
3. Read, report, plot, and export operations consume cached state and never
   choose or launch an analysis.
4. Named method arguments override `options`, stored configuration, and
   defaults in that order; one-call overrides do not mutate stored config.
5. Configuration stores numerical and engineering defaults, never which core
   method runs.
6. The canonical manifest contains every live public operation and every row
   names an existing executable tutorial owner.
7. Removed workflow spellings and frontend-oriented variant APIs are absent
   from supported guides and notebook code.
8. Operation coverage and profile execution coverage are distinct measures.
