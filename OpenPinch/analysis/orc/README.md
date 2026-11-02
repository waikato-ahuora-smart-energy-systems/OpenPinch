# Organic Rankine Cycle Targeting

## Overview
Targets an organic Rankine cycle (ORC) that takes the process surplus below the pinch, using the Direct Integration problem table of a zone. Two models are offered: a Carnot model, and a simulated subcritical cycle solved with CoolProp. Both pick parallel units by multistart minimisation of the annual cost change, using `optimisation` for the search. The result is a `DirectOrcTarget` with the ORC graphs and utilities re-targeted against the residual GCC.

## Interface
- `compute_direct_orc_target(zone, args)` (`service.py`): returns a `DirectOrcTarget`, or `None` when there is no surplus below the pinch; raises `OrcTargetingError` when the surplus is too cold for an ORC or no design saves enough, and for `simulated` when no configured fluid fits.
- `orc_target_inputs(config, args)` (`service.py`): collects `OrcTargetInputs` from a zone `Configuration`.
- `orc_residual_gcc`, `orc_graphs` (`service.py`): GCC with the evaporators as sinks, and the `GCC_ORC` and `NLP_ORC` tables.
- `OrcTargetInputs` (`inputs.py`): frozen, validated settings (kW, degC, K, $/MWh, $).
- `OrcHeatSource`, `orc_heat_source_from_gcc`, `share_at` (`profile.py`): GCC below the pinch and the capacity it offers an evaporator.
- `OrcDesign`, `carnot_orc_design`, `carnot_orc_work`, `orc_temperature_window` (`carnot.py`): design record and the Carnot decision-vector decoder.
- `OrcCycle`, `OrcCycleError`, `solve_orc_cycle`, `critical_temperature` (`cycle.py`): CoolProp cycle state solution.
- `simulated_orc_design`, `simulated_orc_window`, `evaluate_simulated_orc`, `optimise_simulated_orc` (`simulated.py`): simulated model.
- `OrcCosts`, `orc_costs`, `orc_machine_capital_costs` (`costing.py`): installed capital and annual cost change.
- `OrcSearchResult`, `OrcTargetingError`, `search_orc`, `optimise_carnot_orc`, `initial_points`, `require_beneficial` (`search.py`): search and the beneficial-design check.
- Subdirectories: none.

## Invariants
- Units: heat in kW, temperatures in degC, capital in $, costs in $/y; `capital_recovery_factor` is 1/y and is zero when `COSTING_ORC_CAPITAL_RECOVERY_ENABLED` is disabled. [inputs.py:OrcTargetInputs] [service.py:orc_target_inputs]
- `OrcTargetInputs` rejects fewer than one stage, `eta_ii` outside (0, 1], `load_fraction` outside [0, 1], no fluids, and negative `dt_cont`, `min_lift`, `dt_phase_change`, `annual_hours` or `max_superheat`. [inputs.py:OrcTargetInputs.__post_init__]
- `OrcHeatSource` temperatures must not rise, and heat is clamped at zero. [profile.py:OrcHeatSource.__post_init__]
- The pinch is the coldest zero of the GCC; a surplus is required or the source is `None`. [profile.py:orc_heat_source_from_gcc]
- Designs never take more than the cascaded heat available, so no extra hot utility is needed. [carnot.py:carnot_orc_design] [simulated.py:simulated_orc_design] [profile.py:OrcHeatSource.max_sink_duty]
- Carnot net power is `eta_ii * (1 - T_cond/T_evap) * Q_in` on absolute temperatures. [carnot.py:carnot_orc_work]
- The decision vector lies in [0, 1]: 2 variables per unit for Carnot, 3 for simulated; a wrong length raises `ValueError`. [carnot.py:carnot_orc_design] [simulated.py:simulated_orc_design]
- Evaporation stays 5 K below the fluid's critical temperature, and the evaporating temperature must exceed the condensing one. [cycle.py:CRITICAL_MARGIN] [cycle.py:solve_orc_cycle]
- A design counts as beneficial only if it saves at least 1 $/y; otherwise `OrcTargetingError`. [search.py:require_beneficial]
- The simulated search is seeded from the Carnot design when it exists. [service.py:compute_direct_orc_target] [simulated.py:_seed_from_carnot]
- `ORC_MODEL` must be `"carnot"` or `"simulated"`, otherwise `ValueError`, which is raised only after a surplus is found (with no surplus the result is `None`); `OrcTargetingError` and `OrcCycleError` also subclass `ValueError`. [service.py:compute_direct_orc_target]
- The residual GCC is lifted so it is non-negative before utilities are re-targeted. [service.py:_residual_utility_summary]

## Side Effects
- Runs the optimiser backend selected by `bb_minimiser`, which can be slow and is seeded by `OrcTargetInputs.seed`. [search.py:search_orc]
- Works on a deep copy of the base target's problem table; the base target is not changed. [service.py:compute_direct_orc_target]
- CoolProp states are cached per fluid and per thread, and critical temperatures are cached with `lru_cache`. [cycle.py:_state] [cycle.py:critical_temperature]
- CoolProp is imported lazily inside the cycle solver. [cycle.py:_sat]

## Verification
- Run: `uv run --no-sync pytest tests/analysis/orc`
