# HPR implementation units

Approved under the user's authorization to continue through completion.
One distribution, four sequential logical development units; no new services.

1. Accounting (R1/R3/R5; stories 1/2/4): domain load/residual values, load validation,
   Carnot objective and ambient accounting, finite residual cascades. Handoff:
   immutable numerical residual in explicit coordinates and physical boundary.
   Functional Design and Code Generation apply. Exit: balance and duty regressions
   plus generated range/oracle properties and shared HPR tests.
2. Graphs (R4; story 3): analysis graph inventory, target selection and presentation
   transport. Depends on unit 1. Functional Design and Code Generation apply.
   Exit: nonempty HP/RF curves, preserved metadata, explicit selection and repeated
   observation checks.
3. Residual utility workflow (R6; story 5): canonical residual input, accessor,
   allocation dispatch, serialization, frozen utility-placement evaluation.
   Depends on unit 1 and completes after unit 2. Functional Design and Code
   Generation apply. Exit: residual allocation reconciliation, round trips,
   candidate basis preservation and state sequence checks.
4. Tutorial (R2/R7; story 6 and story 2 demonstrations): notebook 08 generator,
   notebook, guides and public inventory. Depends on units 1-3. Functional Design
   is N/A; Code Generation applies. Exit: clean notebook execution and integrated
   checks. Preserve all unrelated notebook changes.

Domain owns immutable values; analysis computes; application owns lifecycle and
provenance; presentation renders. No joint HPR/utility optimization. NFR and
infrastructure stages are skipped as approved; no new technology/deployment.
