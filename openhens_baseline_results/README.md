# OpenHENS Baseline Comparison

Generated from full OpenHENS synthesis runs on branch snapshots, not from checked-in result workbooks.

## Branch snapshots

- main: fbf237bd3bdfbd8cf32698a137f68c0f0db5c1e6
- refactor base: 92e942fec148d5e6a1e052bb3d207b95a4f85379
- refactor current HEAD: 92e942fec148d5e6a1e052bb3d207b95a4f85379

## Case results

| Case | Metric | main | refactor | Delta | Exact match | Within tolerance |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Four-stream-Yee-and-Grossmann-1990-1 | best_solution | 154853.8518602861 | 154853.8518602861 | 0 | True | True |
| Four-stream-Yee-and-Grossmann-1990-1 | solved_esm_count | 100 | 100 | 0 | True | True |
| Four-stream-Yee-and-Grossmann-1990-1 | best_dTmin | 14 | 14 | 0 | True | True |
| Four-stream-Yee-and-Grossmann-1990-1 | best_min_dQ | 0.5 | 0.5 | 0 | True | True |
| Four-stream-Yee-and-Grossmann-1990-1 | best_stages | 3 | 3 | 0 | True | True |
| Four-stream-Yee-and-Grossmann-1990-1 | best_recovery_units | 3 | 3 | 0 | True | True |
| Four-stream-Yee-and-Grossmann-1990-1 | best_cu_units | 2 | 2 | 0 | True | True |
| Four-stream-Yee-and-Grossmann-1990-1 | best_hu_units | 1 | 1 | 0 | True | True |
| Four-stream-Yee-and-Grossmann-1990-1 | total_cases_attempted | 1210 | 1210 | 0 | True | True |
| Four-stream-Yee-and-Grossmann-1990-1 | total_cases_solved | 1210 | 1210 | 0 | True | True |
| Nine-stream-Linnhoff-and-Ahmad-1999-1 | best_solution | 2905807.275299348 | 2905807.275299348 | 0 | True | True |
| Nine-stream-Linnhoff-and-Ahmad-1999-1 | solved_esm_count | 71 | 71 | 0 | True | True |
| Nine-stream-Linnhoff-and-Ahmad-1999-1 | best_dTmin | 18 | 18 | 0 | True | True |
| Nine-stream-Linnhoff-and-Ahmad-1999-1 | best_min_dQ | 1.7 | 1.7 | 0 | True | True |
| Nine-stream-Linnhoff-and-Ahmad-1999-1 | best_stages | 4 | 4 | 0 | True | True |
| Nine-stream-Linnhoff-and-Ahmad-1999-1 | best_recovery_units | 11 | 11 | 0 | True | True |
| Nine-stream-Linnhoff-and-Ahmad-1999-1 | best_cu_units | 3 | 3 | 0 | True | True |
| Nine-stream-Linnhoff-and-Ahmad-1999-1 | best_hu_units | 3 | 3 | 0 | True | True |
| Nine-stream-Linnhoff-and-Ahmad-1999-1 | total_cases_attempted | 1155 | 1155 | 0 | True | True |
| Nine-stream-Linnhoff-and-Ahmad-1999-1 | total_cases_solved | 886 | 886 | 0 | True | True |

## Conclusion

After restoring the downstream parent-problem chain in the refactor and using ESM-only semantics for run summaries, the refactor reproduces the main-branch synthesis values for both benchmark cases within the configured small tolerance. Best-solution values, solved ESM row counts, and best-solution topology values match exactly; a few quartile fields differ only by final-decimal CSV serialization.

The earlier 9-stream mismatch came from downstream TDM/ESM problems being rebuilt without the solved parent problem object, which removed the legacy warm start. The 9-stream case is sensitive to that initialization path, so it selected a different local feasible solution before the fix.

Main was run from a Python 3.12 temporary virtualenv because the legacy main branch fails under Python 3.14 at OrganiseArray/filter((None).__ne__, ...). Refactor was run with the OpenHENS project virtualenv. Both runs used the same case CSVs, min dT/min dQ grids, stage_selection=automated, tolerance=1e-3, best_solns_to_save=10, and max_parallel=10.
