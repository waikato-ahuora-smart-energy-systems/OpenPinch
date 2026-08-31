# Utility Placement Four-Isothermal Notebook Contract Plan

The user's instruction changes the executable notebook contract from two
isothermal plus two sensible levels to four isothermal plus zero sensible
levels for both Process and Site demonstrations. The current executed notebook
and its outputs are preserved.

## TDD checklist

- [x] **Step 1 - RED.** Run the complete suite and capture the exact notebook
  source-contract and generator-equivalence failures.
- [x] **Step 2 - GREEN contract.** Change the notebook test to require two
  occurrences each of `isothermal=4` and `sensible=0`.
- [x] **Step 3 - GREEN ownership.** Align the canonical generator, notebook
  study question, requirements-owned tutorial contract, and RTD workflow while
  retaining the current notebook's executed result cells.
- [x] **Step 4 - Focused verification.** Run notebook packaging, generator,
  documentation consistency, Ruff, and Sphinx checks.
- [x] **Step 5 - Complete verification.** Run the no-deselection configured-
  solver suite and review every changed file for necessity.

Property-Based Testing is N/A: this amendment changes fixed tutorial literals
and generator equivalence, not a generated input domain or algorithmic
property. Security and Resiliency remain disabled and N/A.
