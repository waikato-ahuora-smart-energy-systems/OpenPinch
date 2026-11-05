# Analysis

## Overview
Deterministic engineering calculations that sit between the domain model (`OpenPinch/domain`, `OpenPinch/contracts`) and the application layer. Nothing here imports `application` or `presentation`; `heat_pumps` imports `adapters.optional_dependencies` only. The package holds shared numerical, costing and heat-exchanger helpers in its own files, and the analyses themselves in the subdirectories below. The application layer calls the services in these packages.

## Interface
- `compute_capital_recovery_factor`, `compute_capital_cost`, `compute_annual_capital_cost`, `compute_annual_energy_cost` (`economics.py`): annualisation and cost correlations; `compute_capital_recovery_factor` returns a plain `float` (1/y) and the other three return `Value` objects with `$` or `$/y`.
- `compute_LMTD_from_dts`, `compute_LMTD_from_ts` (`heat_transfer.py`): counterflow log-mean temperature difference.
- `HX_Eff`, `HX_NTU`, `CalcAreaUE`, `MultiPassEff`, `MultiPassNTU`, `CrossflowUnmixedEff1`, `CrossflowUnmixedEff2`, `HX_NTU_Numerical`, `eNTU_slope_Numerical`, `Coth` (`heat_transfer.py`): effectiveness-NTU relations for the `HeatExchangerTypes` arrangements.
- `get_period_index`, `delta_vals`, `delta_with_zero_at_start`, `linear_interpolation`, `interp_with_plateaus`, `make_monotonic`, `g_ineq_penalty` (`numerics.py`): period resolution, differences, interpolation and penalty helpers.
- `supported_workers(requested)` (`parallelism.py`): returns 1 on Emscripten, otherwise the requested worker count.
- Subdirectories:
  - [energy_transfer](./energy_transfer/README.md): energy transfer diagram and surplus/deficit table.
  - [exergy](./exergy/README.md): exergy curves and exergy transfer effectiveness on existing targets.
  - [graphs](./graphs/README.md): serialised graph data built from target tables.
  - [heat_exchanger_networks](./heat_exchanger_networks/README.md): heat exchanger network synthesis.
  - [heat_pumps](./heat_pumps/README.md): heat pump and refrigeration targeting.
  - [orc](./orc/README.md): organic Rankine cycle targeting.
  - [power](./power/README.md): steam turbine cogeneration targeting.
  - [targeting](./targeting/README.md): problem table, GCC, utility, area and Total Site targeting.
  - [thermodynamics](./thermodynamics/README.md): water and steam property wrappers.
  - [utility_placement](./utility_placement/README.md): utility placement optimisation.

## Invariants
- `get_period_index` resolves `period_id` or `period_idx` to one index, defaults to 0, and raises `ValueError` for a negative index, for a conflicting id and index pair whether or not period ids are registered (an id with none registered resolves to 0), and, when the collection has period ids, for an unknown id or index; a boolean or float index raises `TypeError` and a non-integer string raises `ValueError`. [numerics.py:get_period_index]
- Differences within `tol` of zero are set to exactly 0.0. [numerics.py:delta_vals]
- The capital recovery factor is `1/n` at a zero rate, and rejects a non-positive life or a rate at or below -1. [economics.py:compute_capital_recovery_factor]
- Annualised capital cost requires a discount rate of at least 0 and a service life of at least 1 year, and returns 0 for capital within `tol` of zero. [economics.py:compute_annual_capital_cost]
- Capital cost is zero for area within `tol` of zero, uses at least one unit, and clamps negative coefficients to zero. [economics.py:compute_capital_cost]
- `compute_annual_energy_cost` clamps negative power, price and operating hours to zero. [economics.py:compute_annual_energy_cost]
- The LMTD requires strictly positive end differences (rounded to 6 dp); `compute_LMTD_from_ts` also raises when the hot inlet is below the hot outlet or the cold outlet is below the cold inlet (equal temperatures pass). [heat_transfer.py:compute_LMTD_from_dts] [heat_transfer.py:compute_LMTD_from_ts]
- `CalcAreaUE` raises `ValueError` for infeasible heat flow direction or effectiveness above one, and returns infinity at unit effectiveness. [heat_transfer.py:CalcAreaUE]
- An unrecognised exchanger arrangement is treated as counter flow in `HX_Eff` and gives `-Passes` in `HX_NTU` (`-1` for a single pass) only when the multipass-converted effectiveness is in `(0, 1)` (otherwise `0`). [heat_transfer.py:HX_Eff] [heat_transfer.py:HX_NTU]
- `g_ineq_penalty` requires a `PenaltyForm` member. [numerics.py:g_ineq_penalty]

## Side Effects
None. The package's own files do no IO and keep no caches or global state.

## Verification
- Run: `uv run --no-sync pytest tests/analysis/test_economics.py tests/analysis/test_heat_transfer.py tests/analysis/test_parallelism.py tests/analysis/test_interpolator.py`
