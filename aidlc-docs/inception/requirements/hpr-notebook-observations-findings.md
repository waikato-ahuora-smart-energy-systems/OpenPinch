# HPR notebook observations: investigation

Date: 2026-09-07. Status: investigation complete; implementation not started.

## Investigation checklist

- [x] Load workflow rules, previous architecture and extension configuration.
- [x] Inspect notebook 08, including local source changes and saved outputs.
- [x] Reproduce plotting selection and compare actual target outputs.
- [x] Trace penalties, configured prices, load selection and ambient handling.
- [x] Run cold-price and load-fraction sensitivity cases.
- [x] Trace residual utility allocation and utility-placement input ownership.
- [x] Run existing affected tests and document proposed requirements.

## 1. Identical results versus identical plots

The current notebook compares a direct process heat pump with utility-system
refrigeration. These have different background temperatures; the heat pump
explicitly uses one condenser and evaporator, while the refrigeration call
falls back to three condensers and two evaporators. This is not a controlled
comparison of the two operating modes.

A fresh execution of the notebook's two target calls produced different
results: approximately 373.489 kW heat-pump work and 484.438 kW refrigeration
work. The repeated net-load plot is nevertheless identical because
`_get_hpr_graphs()` returns an empty dictionary for refrigeration
(`OpenPinch/analysis/heat_pumps/service.py:401`). The plot accessor consequently
finds the earlier heat-pump result. There is no target selector on the named
plot methods to make this distinction explicit.

The saved notebook also contains a `NameError` for `load_rfgn_plot` in the
display cell. Its saved outputs therefore do not establish that both targeting
cells completed in that kernel session. The summary variable is captured
before refrigeration runs and does not include its result.

## 2. Empty grand composite curve

`OpenPinch/analysis/targeting/direct.py:652` emits a GCC-with-heat-pump record
for the ordinary direct-integration target even though its HPR columns contain
no solved data. The graph accessor chooses the first matching graph type.

After the notebook's heat-pump call:

| Call | Result |
|---|---|
| `problem.plot.net_load_profiles_with_heat_pump()` | 4 traces |
| `problem.plot.grand_composite_curve_with_heat_pump()` | 0 traces |
| `problem.plot.grand_composite_curve_with_heat_pump(index=1)` | 9 traces |

The indexed call is a diagnostic workaround for this exact inventory, not a
durable selection contract. Refrigeration does not currently emit its own GCC.
Graph transport also discards graph names and segment metadata, reducing the
ability to identify selected outputs; `Graph` and `Segment` in
`OpenPinch/contracts/graphs.py` omit fields supplied by the graph builder.

## 3. Penalties and prices

`build_hpr_accounting()` only penalizes residual cold duty when
`is_heat_pumping` is false. With work 10, residual cold duty 100, selected load
100, cold-price ratio zero and no cycle constraint violations, a direct
accounting probe returned:

| Mode | Feasibility penalty | Objective |
|---|---:|---:|
| Heat pump | 0 | 0.1 |
| Refrigeration | 100000 | 1000.1 |

There is no heat-pump penalty to remove for simply leaving low-grade heat
unused. Existing allocation penalties address attempted infeasible duties.

Carnot targeting uses:

`(work + residual_heat * heat_price_ratio + residual_cold * cold_price_ratio + penalty) / selected_load`

Both `COSTING_HPR_PRICE_RATIO_HEAT_TO_ELE` and
`COSTING_HPR_PRICE_RATIO_COLD_TO_ELE` default to 1.0. The sample's HU and CU
prices are both 10 $/MWh, but those stream prices are used in utility reporting,
not to derive the Carnot objective ratios. Lowering only the sample CU price
does not correct the illustrative Carnot economics.

The following public-call sensitivity used separate problems, one condenser,
one evaporator, one restart, direct placement and the same sample input.
Heat/electricity ratio remained 1.0. Results are observed solver outputs,
not guaranteed global optima or validated plant operating points.

| Mode | Fraction | Cold/electricity ratio | Condenser kW | Evaporator kW | Work kW | Ambient source kW | Ambient sink kW |
|---|---:|---:|---:|---:|---:|---:|---:|
| Heat pump | 0.25 | 1.0 | 1373.489 | 1000.000 | 373.489 | 0 | 1185.989 |
| Heat pump | 0.25 | 0.1 | 187.500 | 165.649 | 21.851 | 0 | 0.000006 |
| Heat pump | 0.25 | 0 | 187.500 | 165.649 | 21.851 | 977.624 | 0 |
| Refrigeration | 0.25 | 0 | 697.546 | 250.000 | 447.546 | 0 | 0 |
| Heat pump | 1.0 | 0 | 750.000 | 489.908 | 260.092 | 496.609 | 0 |
| Refrigeration | 1.0 | 0 | 0 | 0 | 0 | 0 | 1000.000 |

Changing the ratio to 0.1 leaves most process source heat unused. This confirms
that default cooling economics, rather than a heat-pump cooling penalty,
drive the full-source-use example. Zero-price cases expose unconstrained or
economically indifferent ambient exchanges and are not a sufficient repair.
The zero-work full-load refrigeration result also needs an explicit distinction
between direct ambient cooling and refrigeration duty before being presented
as an equipment target.

Reproduction pattern:

```python
from OpenPinch import PinchProblem

problem = PinchProblem("heat_pump_targeting.json")
heat_pump = problem.target.carnot_heat_pump(
    load_fraction=0.25,
    condensers=1,
    evaporators=1,
    maximum_restarts=1,
    options={"COSTING_HPR_PRICE_RATIO_COLD_TO_ELE": 0.1},
)
print(heat_pump.hpr_details)
print(problem.summary_frame())
```

For execution as a standalone script, place solver calls inside an
`if __name__ == "__main__":` guard because the optimizer uses multiprocessing.
An initial stdin probe triggered worker-spawn errors; the reported graph and
sensitivity results were subsequently reproduced using guarded script files.

## 4. Meaning of load_fraction

`resolve_hpr_target_load()` multiplies the maximum absolute selected background
profile ordinate by the fraction. Heating uses the process cold/demand profile;
refrigeration uses the process hot/source profile. Utility-system placement
uses profiles from inverted utilities instead of direct process profiles.

In this sample, direct integration has 750 kW residual heating and 1000 kW
residual cooling. Therefore 0.25 selects 187.5 kW of heating or 250 kW of cooling.
Preprocessing trims the relevant profile to that duty, starting from its
low-load/pinch end. The opposite profile remains available in full.

This is neither compressor part-load nor a guaranteed delivered duty. The
optimizer still chooses duty within the selected background, and
`Q_heat_base = x_heat_base * (Q_heat_max + Q_amb_cold)` allows ambient sink duty
to expand total condenser duty beyond the selected process heating. Values
greater than one are also currently accepted; fraction selection is not capped
in the same way as `load_duty`.

## 5. Residual utility targeting and additional numerical findings

Both HPR service paths already call `_get_hpr_residual_utility_summary()`.
It subtracts the HPR curve from the relevant background curve, removes GCC
pockets, separates load profiles and retargets copies of the existing utility
collections. Returned HPR target utility collections and utility targets are
intended to be the leftover duties; another base heat-integration call is not
needed to obtain them.

However, the real sample exposes unresolved consistency problems:

- At cold-price ratio 0.1, approximately 187.5 kW condenser duty and 165.649 kW
  evaporator duty coexist with reported residual utilities of 749.900 kW hot
  and 1021.751 kW cold. At ratio zero, essentially the same cycle temperatures
  and duties give 562.500 kW hot and 834.351 kW cold, with a large unused
  ambient source also reported. Full temperature-profile accounting, ambient
  treatment and numerical boundary handling need regression oracles; simple
  scalar energy balance alone is insufficient to establish correctness.
- In the zero-cold-price 25% heat-pump run, both stored after-HPR net-load
  columns contain only NaN values. `_get_hpr_residual_load_profiles()` stores
  them only when pocket removal leaves the same temperature grid. New pocket
  breakpoints bypass that update (`common/postprocessing.py:194`).

`target.utility_placement()` cannot currently optimize this selected HPR
residual. `_extract_period()` reconstructs a problem from source input and
executes direct/indirect/Total-Site heat integration. Candidate replay does
the same. HPR target snapshots are not consumed, so calling utility placement
after HPR does not imply chaining onto its residual.

A useful extension is sequential: freeze the chosen HPR duties and ambient
exchange, then allocate or optimize utilities on its residual profiles. Joint
optimization of HPR equipment and utility levels is a separate larger problem.

## Verification

57 existing targeting, cascade/parallel Carnot and simple-graph tests passed.
They do not cover the observed full public-workflow failures. An initial test
invocation with `--no-cov` was rejected because this environment does not load
that pytest option; the plain configured invocation passed.

No application code, sample data or notebooks were changed during investigation.
Existing notebook edits and execution outputs were preserved.
