# HEN Median Design Basis

## Overview
Builds the synthetic single-period case a multiperiod HEN is synthesised from on the median design basis. It works on problem data (`PinchProblem.to_problem_json()`) with plain numpy, so it runs in Pyodide; `../../../application/_problem/accessors/design.py` turns the case into a `PinchProblem`, synthesises on it and fits the network to every period.

## Interface
- `weighted_k_medoids`, `Clustering` (`clustering.py`): weighted k-medoids (PAM: build, then swap) over the rows of a feature matrix; medoid rows in build order and each row's cluster.
- `median_basis`, `MedianBasis`, `BasisCluster`, `BasisStream` (`synthetic_case.py`): cluster a problem's operating periods and build each cluster's synthetic case (problem data) and stream table; `basis_cluster` is the heaviest cluster.
- `period_source` (`synthetic_case.py`): problem data with every per-period quantity reduced to one period's value; also used by `single_period_case`.
- `MEDIAN_QUANTITIES`, `SYNTHETIC_PERIOD_ID` (`synthetic_case.py`): the stream quantities that take a weighted median, and the synthetic period's id, `median`.
- Subdirectories: none.

## Invariants
- A medoid minimises the sum of weight times Euclidean distance over its members. Build adds, `k` times, the row lowering that cost most; swap then takes the best cost-lowering replacement of a medoid each pass, until none lowers it by more than 1e-12 of the cost. Ties go to the lowest row, and rows join the nearest medoid, a tie to the earlier medoid; `k` lies in 1..rows [clustering.py:weighted_k_medoids].
- Scaling every weight by one constant, or duplicating a period instead of doubling its weight, gives the same clusters; the period order changes nothing beyond tie-breaks [clustering.py:weighted_k_medoids].
- Operating periods have weight above zero and some process stream running (active, heat flow above zero); only they are clustered, and `basis_clusters` lies in 1..their number [synthetic_case.py:median_basis].
- Features per stream are supply and target temperature, heat flow and the running flag, each divided by its period-weighted standard deviation over the operating periods (a column with no spread is left as it is) [synthetic_case.py:_scaled_features].
- A cluster's case: per stream, the lower period-weighted median of `t_supply`, `t_target`, `heat_flow`, `htc` and `dt_cont` over the member periods it runs in (a quantity some member lacks keeps the medoid's value), with `heat_capacity_flowrate` cleared so it follows; a stream running in less than `active_share` of the cluster's weight is left out. Utilities and options come from the medoid period, with one period, `median`, of weight 1 [synthetic_case.py:_cluster_case].
- Medians keep each stream's direction: in every period a hot stream's supply is above its target, so with the same weights over the same periods the median supply is too.
- Streams with segments or a profile are rejected [synthetic_case.py:median_basis].
- The heaviest cluster is the basis cluster, a tie going to the cluster whose medoid comes first [synthetic_case.py:median_basis].

## Side Effects
- None: reads problem data and returns new data.

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks/test_basis_clustering.py tests/application/test_hen_design_basis.py`
