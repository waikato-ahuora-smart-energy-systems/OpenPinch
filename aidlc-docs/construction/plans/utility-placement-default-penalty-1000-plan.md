# Utility Placement Default-Utility Penalty 1000 Plan

The private `HU`/`CU` fallback penalty coefficient increases from 100 to 1000.
This is ten times the immediately preceding coefficient and one hundred times
the original coefficient of 10. Physical entropy and feasibility are unchanged.

## TDD checklist

- [x] **Step 1 - Oracle-first verification.** Change fixed, integration,
  application, and generated oracle expectations from coefficient 100 to
  coefficient 1000. The working implementation was already 1000 when the gate
  ran, so it passed immediately; no artificial regression was introduced.
- [x] **Step 2 - GREEN.** Retain the single private Utility Placement
  coefficient at 1000 without modifying the generic numerical kernel.
- [x] **Step 3 - Documentation.** Update requirements and RTD equations while
  preserving dimensionless reporting and weighted-period aggregation.
- [x] **Step 4 - Focused verification.** Run penalty properties, evaluation,
  public integration, notebook contracts, Ruff, and Sphinx.
- [x] **Step 5 - Complete verification.** Run the no-deselection configured-
  solver suite and review every changed file for necessity.

Property-Based Testing remains enabled through the coefficient-1000 canonical
oracle and scaling properties. Security and Resiliency remain disabled and N/A.
