# HPR correction persona

## P1: Process Engineer

An engineer uses pinch analysis to screen heat pumps and refrigeration, compare
utility demand and choose promising operating conditions before detailed
equipment selection. Python experience varies; understanding package internals
must not be necessary to run or interpret the public workflow.

This is the existing OpenPinch Process Engineer persona, focused on the approved
HPR correction. It does not introduce a new role or permission model.

### Goals

- Choose a meaningful heating or cooling service duty and distinguish it from
  compressor part-load and total cycle duty.
- Compare economic scenarios using explicit assumptions and consistent inputs.
- Inspect curves and summaries belonging to the intended target and period.
- Understand the remaining thermal loads, ambient exchange and utility demand.
- Optimize utility placement after fixing an HPR target without changing that
  target or losing the original study.
- Reproduce and adapt the packaged notebook using public operations.

### Current obstacles

Notebook 08 can show an empty GCC and reuse heat-pump plots after refrigeration.
The selected fraction, ambient duty and achieved process duty are difficult to
distinguish. Separate price settings obscure the objective. Missing or
inconsistent residual profiles prevent a trustworthy follow-on utility study.

### Story mapping

| Story | Persona goal |
|---|---|
| HPR-US-1 | Select and interpret service duty. |
| HPR-US-2 | Compare economics and operating modes. |
| HPR-US-3 | Inspect the intended target's curves. |
| HPR-US-4 | Reconcile leftover loads and utilities. |
| HPR-US-5 | Optimize utilities against the chosen residual. |
| HPR-US-6 | Reproduce and adapt the full notebook workflow. |
