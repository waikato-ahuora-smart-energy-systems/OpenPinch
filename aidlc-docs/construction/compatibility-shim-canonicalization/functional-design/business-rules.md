# Business Rules

1. No removed runtime name is accepted through property access, construction,
   reflection, helper strings, imports, serialization aliases, or forwarding.
2. Compact JSON keys are wire vocabulary only and never reappear as public runtime
   properties.
3. Unknown external input fields are errors rather than ignored data.
4. A workspace bundle without exactly schema version `3` is invalid.
5. Summary detail is a boolean choice; numeric comparison formatting is private.
6. Graph kind is selected through named methods and full enum identities.
7. A design view exposes only documented explicit members; all result fields are
   accessed through `view.result`.
8. Friendly normalization retained by the approved plan is not treated as legacy
   compatibility.
9. Numerical equations, solver fallbacks, and the required parent-axis placeholder do
   not change.

