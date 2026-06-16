# HENS-09 Adversarial Re-Review

## Findings

No findings. HENS-09 is cleared.

## Prior Finding Resolution

- The prior blocker on the public `problem.design.heat_exchanger_network_synthesis(...)` options bypass is resolved. The design accessor now validates the raw `options` object with `_normalise_runtime_options(...)` before adding `state_id`, so `TargetInput` and iterable option-like objects are rejected before they can be coerced into a plain dict (`OpenPinch/classes/_problem/_design_accessor.py:18`, `OpenPinch/classes/_problem/_design_accessor.py:31`).
- The shared service guard still rejects non-dict runtime options and direct `HENS_*` runtime overrides, keeping persistent HEN controls rooted in `TargetInput.options` / prepared `Configuration` (`OpenPinch/services/heat_exchanger_network_synthesis/service.py:79`, `OpenPinch/services/heat_exchanger_network_synthesis/service.py:89`).
- Public-path negative tests now cover `TargetInput` as options, the iterable object shape that the old `dict(options)` coercion accepted, and case/study-like objects (`tests/test_heat_exchanger_network_public_service.py:90`, `tests/test_heat_exchanger_network_public_service.py:174`).
- I confirmed the previous smoke case now raises `TypeError` instead of returning successfully, and the iterable/case/study option objects also raise `TypeError`.

## Scope And Contract Checks

- HENS-09 remains limited to public docs, public-surface tests, conservative reference exports, and guard hardening. I did not see solver movement, helper replacement, regression-tier expansion, OpenHENS retirement, public raw-input runner, or root service export in the reviewed diff.
- Public examples continue to start from `PinchProblem` / `PinchWorkspace` with OpenPinch-compatible JSON or native `TargetInput`, and the guide states source OpenHENS CSV files are migration source material only (`docs/guides/heat-exchanger-network-synthesis.rst:4`, `docs/guides/heat-exchanger-network-synthesis.rst:43`, `docs/guides/heat-exchanger-network-synthesis.rst:94`).
- The internal service is documented only as an internal implementation boundary, while user-facing paths are the design accessor and workspace dispatch (`docs/guides/heat-exchanger-network-synthesis.rst:12`).
- Public exports remain OpenPinch-native. The tests cover absence of an `OpenHENS` facade, import shim, command parity contract, root-exported service, public case/study roots, raw-input runner, and OpenHENS field aliases (`tests/test_heat_exchanger_network_public_service.py:34`, `tests/test_heat_exchanger_network_public_service.py:95`, `tests/test_heat_exchanger_network_public_service.py:109`, `tests/test_heat_exchanger_network_public_service.py:129`, `tests/test_heat_exchanger_network_public_service.py:192`).
- The guide documents `TargetOutput.design`, `HeatExchangerNetwork` source/sink links, optional exports from `problem.results`, optional dependency installation, missing-solver expectations, and marked synthesis/solver test commands (`docs/guides/heat-exchanger-network-synthesis.rst:121`, `docs/guides/heat-exchanger-network-synthesis.rst:136`, `docs/guides/heat-exchanger-network-synthesis.rst:194`).
- The OpenHENS-to-OpenPinch mapping is documentation-only and explicitly disclaims runtime aliases, OpenHENS field aliases, command parity, and an `OpenHENS` facade (`docs/guides/heat-exchanger-network-synthesis.rst:156`).
- Docs and comments reviewed in this pass describe stable public contracts, ownership boundaries, and migration constraints without introducing implementation-trivia commentary or misleading compatibility promises.
- The HENS-09 Definition of Done remains review-dependent and unchecked in the task file, which is appropriate for an implementation worker to update after review clearance (`docs/developer/openhens-integration-tasks/09-public-service-and-documentation.md:106`).

## Residual Risks

- I did not rerun the full pytest/docs/ruff commands because this review is read-only and those commands can write caches or build artifacts. I relied on the implementation notes' recorded results plus targeted no-bytecode smoke checks for the prior bypass.
- Live solver-backed synthesis was not exercised in this review. That remains covered by marked solver-test expectations and later solver/regression gates rather than HENS-09's public docs/API slice.
- `.DS_Store` remains dirty in the worktree and outside HENS-09 scope; it should stay excluded from staging/publication (`docs/developer/openhens-integration-tasks/09-public-service-and-documentation.md:176`).

## Review Status

Cleared. The prior blocking public-path bypass is resolved, the added negative coverage exercises the failure mode directly, and I found no remaining HENS-09 blockers.
