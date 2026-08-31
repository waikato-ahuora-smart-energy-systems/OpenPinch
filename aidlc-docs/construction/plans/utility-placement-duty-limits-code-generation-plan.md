# Utility Placement Maximum-Duty Code Generation Plan

## Unit Context

- **Existing owners**: placement contracts/pure model, evaluation/optimization,
  public workflow/notebook integration, and shared utility targeting.
- **Approved API**: optional `maximum_duties` mapping by globally unique name.
- **Approved behavior**: per-period caps, omitted unbounded, zero disabled,
  residual-only visible fallback, and squared `g_penalty()` ranking.
- **Exclusions**: no monetary placement, CLI, new plot API, dependency, or
  infrastructure change.
- **Property extension**: PBT-01 through PBT-10 remain enabled and blocking.

## TDD Execution Steps

- [x] **Step 1 - Requirements, workflow, and functional design.** Record the
  approved mapping, capacity constraints, fallback semantics, penalty equation,
  persistence, acceptance tests, phase decisions, and property inventory.
- [x] **Step 2 - RED: public and contract validation.** Add failing tests for
  the accessor signature, generated/inferred name resolution, scalar/unit and
  period values, unknown names, invalid values, independent pair limits, JSON
  round trips, and batch/all-period forwarding.
- [x] **Step 3 - RED: allocation, fallback, and penalty properties.** Add
  failing example and Hypothesis tests for capped targeting, zero/unbounded
  behavior, named-before-fallback priority, squared penalty identities,
  period-weighted aggregation, and physical-entropy separation.
- [x] **Step 4 - GREEN: capacity contracts and runtime utility metadata.** Add
  typed placement limit evidence plus optional period-aware runtime maximum
  heat flow without changing allocated `heat_flow` semantics.
- [x] **Step 5 - GREEN: cap-aware targeting and residual fallback.** Bound each
  utility assignment, generate deterministic `HU`/`CU` fallback definitions,
  preserve coverage, and keep existing uncapped targeting unchanged.
- [x] **Step 6 - GREEN: evaluation, results, and `g_penalty()`.** Keep fallback
  candidates feasible, include fallback in balanced composites, compute and
  aggregate squared penalties, and expose separate typed evidence.
- [x] **Step 7 - GREEN: public integration and returned-case persistence.** Wire
  `maximum_duties` through problem/workspace/batch surfaces and write caps plus
  required fallback into the detached normal case for standard retargeting.
- [x] **Step 8 - Refactor, notebook, RTD, and summaries.** Remove superseded
  default-forbidden logic, update the canonical notebook generator and executed
  notebook, add one cap/fallback demonstration, update RTD and only affected
  AI-DLC summaries, and retain no cosmetic changes.
- [x] **Step 9 - Complete verification.** Run focused/PBT tests, coverage for
  new numerical kernels, Ruff, patch hygiene, warning-as-error Sphinx, notebook
  generation/execution, source/wheel contents, isolated installed-wheel smoke,
  representative performance tests, and the complete solver-enabled suite.
- [x] **Step 10 - Review and commit.** Review every changed file for necessity,
  preserve unrelated `.gitignore` work, record exact evidence, close state, and
  commit only the maximum-duty amendment to `develop`.

## Production Paths

- `OpenPinch/contracts/input.py`
- `OpenPinch/contracts/utility_placement.py`
- `OpenPinch/domain/stream.py`
- `OpenPinch/application/_problem/input/utilities.py`
- `OpenPinch/analysis/targeting/utilities.py`
- `OpenPinch/analysis/utility_placement/context.py`
- `OpenPinch/analysis/utility_placement/allocation.py`
- `OpenPinch/analysis/utility_placement/penalties.py`
- `OpenPinch/analysis/utility_placement/evaluation.py`
- `OpenPinch/application/utility_placement.py`
- `OpenPinch/application/_problem/accessors/target.py`
- `scripts/generate_tutorial_notebooks.py`
- `OpenPinch/data/notebooks/19_utility_placement_optimisation.ipynb`
- affected RTD utility-placement guide

## Test Paths

- `tests/domain/test_stream.py`
- `tests/analysis/test_direct_targeting.py`
- `tests/analysis/utility_placement/`
- `tests/application/test_utility_placement.py`
- `tests/application/test_utility_placement_batch.py`
- `tests/packaging/test_notebooks.py`
- affected documentation and distribution tests

This plan is the Code Generation source of truth and is approved under the
user's standing authorization through completion unless an unexpected conflict
appears.
