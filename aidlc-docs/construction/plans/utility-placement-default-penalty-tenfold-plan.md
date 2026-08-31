# Utility Placement Default-Utility Penalty Tenfold Plan

The default `HU`/`CU` fallback penalty coefficient increases from 10 to 100.
This is a tenfold ranking change only; physical entropy generation, fallback
duties, feasibility, and result units remain unchanged.

## TDD checklist

- [x] **Step 1 - RED.** Change fixed, integration, application, and generated
  oracle expectations from coefficient 10 to coefficient 100 and confirm the
  current implementation fails by exactly a factor of ten.
- [x] **Step 2 - GREEN.** Pass the explicit utility-placement coefficient 100
  to the shared squared inequality-penalty kernel without changing its generic
  default for other analyses.
- [x] **Step 3 - REFACTOR and documentation.** Keep one local named coefficient,
  update the requirements equation and RTD formula, and preserve dimensionless
  reporting and weighted-period aggregation.
- [x] **Step 4 - Focused verification.** Run penalty properties, evaluation,
  application integration, notebook contracts, Ruff, and warnings-as-errors
  documentation checks.
- [x] **Step 5 - Complete verification.** Run the no-deselection configured-
  solver suite, refresh executable notebook evidence if affected, and review
  every changed file for necessity.

Property-Based Testing remains enabled: the scale-invariance and canonical
oracle properties use coefficient 100. Security and Resiliency remain disabled
and N/A.
