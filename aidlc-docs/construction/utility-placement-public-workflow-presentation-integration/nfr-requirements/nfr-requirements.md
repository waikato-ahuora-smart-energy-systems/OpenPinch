# Unit 3 NFR Requirements

1. Public request validation must finish before base targeting begins.
2. Success and every typed failure must leave source problem JSON, legacy results, selected period, and active workspace case unchanged.
3. Fixed-seed repeated calls must return equal detached results.
4. Explicit period and case sequences must be retained exactly.
5. Batch execution must return every success and failure independently.
6. Observation and reporting must execute no target or optimiser callback.
7. Specialist result JSON round trips must be lossless through the public path.
8. Existing root exports and legacy target result schemas must not expand.
9. The public accessor overhead excluding targeting and optimisation must remain negligible relative to the Unit 2 one-second cold replay gate.
10. Tutorial 19 must execute from source and installed wheel with its declared profile.
11. The tutorial manifest and distribution must contain exactly one utility-placement notebook.
12. Ruff, architecture, focused integration, fixed-seed properties, non-solver regression, documentation, wheel, source, and isolated-import gates must pass.
13. No runtime dependency, external service, credential, database, or deployment resource may be added.
14. Diagnostics shown by reports must remain bounded by the Unit 2 ten-representative contract.
