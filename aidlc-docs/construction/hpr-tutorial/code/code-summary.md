# Notebook and documentation implementation

R2/R7; HPR-US-2, 6. Notebook 08 and its generator now compare HP and refrigeration
on the same process basis, stage counts and explicit prices. The positive cold
ratio is 0.1; global defaults remain 1.0. The interpretation explains the cost
sensitivity separately from feasibility penalties and utility stream prices.
Selected-service semantics, separate summaries, four target-specific plots,
residual allocation and utility placement are demonstrated.

Only notebook 08 was regenerated. Independent edits and saved outputs in other
notebooks and the user's workspace JSON were preserved. Notebook 08 validates
as nbformat and matches its generator. The public method and tutorial inventories
and both HPR/utility-placement guides have been updated.

Fresh Jupyter-kernel execution completed all code cells, four nonempty plots,
residual placement and basis-preservation assertions. Observed HP heating is
187.500 kW (work 21.851 kW); RF cooling is 250.000 kW (work 447.546 kW).
HP residual utilities are 562.500 kW hot and 834.351 kW cold. These are screening
observations, not universal optimum assertions. Strict Sphinx and wheel/sdist
builds pass. Three pre-existing notebook preservation checks are documented in
the integrated verification record.

## Extension compliance

PBT-01 through PBT-10: compliant across the integrated change. Designs identify
properties; generated residual cases exercise the independent offset oracle,
nonnegative/finite profiles and idempotence. Graph and frozen-input JSON round
trips preserve data. Generated operation sequences compare allocation, roundtrip
and price edits with the unchanged reference basis after every operation,
including empty sequences. Constrained Hypothesis generators retain shrinking;
seed 20260715 matches CI. Real solver, graph and segmented-utility examples
complement properties. Commutativity/induction are N/A for these operations.
Security and Resiliency are disabled in aidlc-state.md and were skipped.
