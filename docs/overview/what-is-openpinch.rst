What Is OpenPinch?
==================

OpenPinch is a process-engineering package for Pinch Analysis, Total Site heat
integration, multiperiod analysis, HPR and cogeneration screening, exergy and
energy-transfer analysis, and heat exchanger network synthesis.

The public application model has two classes:

``PinchProblem``
   One prepared case with explicit ``target``, ``design``, ``components``, and
   ``plot`` accessors plus cached result, report, and export operations.

``PinchWorkspace``
   Named cases, unsolved scenario creation, active-case forwarding, ordered
   case batches, comparison, and persistence.

The distinction between execution and observation is deliberate. A named
target or design method performs engineering work. Validation, state
properties, summaries, metrics, reports, plots, comparisons, and exports do not
select or launch an analysis.

Configuration provides reusable numerical and engineering defaults. The method
name selects the workflow. Per-call keywords override ``options``, stored
configuration, and defaults without mutating persistent configuration.

Start with :doc:`../getting-started` and the
:doc:`../examples/notebook-series`.
