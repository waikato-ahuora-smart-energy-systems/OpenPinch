# Prepared Target Replay Summary

## Outcome

Utility-placement candidate evaluation no longer reconstructs and preprocesses
a complete `PinchProblem` for every candidate and period. One detached problem
is prepared per adapter, and its utility-independent direct load profiles are
cached once per period and target zone.

## Replay boundary

For every candidate, the adapter:

1. creates a lightweight zone-tree clone that shares read-only process stream
   collections;
2. resets targets, graphs, derived net streams, utilities, results, and process
   component state;
3. reconstructs only the candidate utility streams;
4. deep-copies the prepared shifted and real process problem tables;
5. inserts the candidate utility temperature endpoints using the existing
   `ProblemTable.insert_temperature_interval` path; and
6. delegates duty allocation, balanced composites, and Process or Total Site
   aggregation to the existing targeting services.

The cache deliberately excludes utilities, duties, balanced composites, and
Total Site aggregation because those values depend on the candidate.

## Equivalence and performance evidence

Fresh-problem targeting remains the independent oracle. Example and generated
tests compare complete problem tables, utility names and duties, heat targets,
Process snapshots, Total Site snapshots, source immutability, repeated replay,
and pickle round trips.

With three warmups and ten measured calls, median uncached replay changed from
0.256325 seconds to 0.028981 seconds for the Process case (8.84 times faster)
and from 0.537118 seconds to 0.324896 seconds for Total Site (1.65 times
faster). A three-generation, four-level end-to-end run changed from 27.861657
seconds to 3.258408 seconds for Process and from 49.468948 seconds to
30.589541 seconds for Total Site.

The remaining Total Site cost is candidate-specific: child utility targeting,
subzone aggregation, net-stream reconstruction, and site target aggregation.
