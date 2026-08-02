# Total Site Profile Hierarchy Fix Build and Test Summary

## Results

| Gate | Result | Status |
|---|---|---|
| Expanded domain, targeting, graph, presentation, orchestration, multi-scale, multi-period, and HPR | 145 passed | Pass |
| Notebook 2 base-profile execution | 1 passed; 9 deselected | Pass |
| Complete fixed-seed non-solver suite | 2,191 passed; 3 skipped; 4 deselected in 172.73 s | Pass |
| Integrated packaging and notebook resources | 84 passed; 3 optional-profile skips | Pass |
| Repository Ruff lint | All checks passed | Pass |
| Repository Ruff format | 461 files already formatted | Pass |
| Isolated wheel and source distribution | OpenPinch 0.5.3 wheel and sdist built | Pass |
| Built-wheel Zone and graph smoke | Independent pairs and rounded SUGCC corner verified from wheel | Pass |
| Patch whitespace | `git diff --check` clean | Pass |

## Notes

The first complete run found transient execution counts and outputs in the
already-modified Notebook 4. Only those transient fields were cleared; notebook
source and metadata improvements were preserved. The two source-only packaging
regressions then passed, and the complete suite was rerun successfully.

The isolated wheel's first import attempt intentionally ran outside the project
environment and demonstrated that dependencies were absent. The accepted smoke
used the locked runtime dependencies while forcing `OpenPinch` and Notebook 2
to load directly from the built wheel path.

After the ownership model was reopened, a regression-first run produced seven
expected failures and 30 passing controls. The second persistent pair was then
added, after which 37 direct regressions, 141 expanded affected tests, and all
complete gates passed. The final wheel smoke constructs `Zone` from the wheel
and verifies that each own-zone collection is distinct from its corresponding
immediate-subzone collection.

The SUGCC correction began with two expected regression failures: the generic
cleanup contract and `pulp_mill.json` graph geometry both showed the missing
138.5254 degC corner. Exact consecutive coordinate deduplication restored the
vertical connector and LPS ledge. Targets and utility duties were unchanged.

## Completion

All functional requirements and acceptance criteria are satisfied. Security
and Resiliency extensions are disabled and N/A. The enabled partial PBT rules
are compliant. Operations is N/A because no deployment change was requested.
