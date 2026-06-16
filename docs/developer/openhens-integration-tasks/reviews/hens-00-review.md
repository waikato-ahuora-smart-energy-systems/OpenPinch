# HENS-00 Adversarial Re-review

## Findings

No blocking findings. HENS-00 is cleared after the implementation worker's fix
pass.

## Prior Finding Resolution

### Resolved: Baseline provenance is now reproducible enough for HENS-00

The prior blocker was that the refactor baseline artifact snapshot recorded a
dirty OpenHENS worktree without a stored patch artifact or hash. That is now
addressed. The task file lists the checked-in provenance patch and `.sha256`
artifact (`docs/developer/openhens-integration-tasks/00-baseline-freeze-and-acceptance-matrix.md:107`),
and the manifest records the patch path, SHA256, and reverse-apply verification
for the OpenHENS dirty artifact source
(`openhens_baseline_results/baseline_manifest.json:49`,
`openhens_baseline_results/baseline_manifest.json:52`). The refactor branch
summary now also points from the dirty recorded artifact snapshot to the stored
patch and clean target head
(`openhens_baseline_results/refactor/branch_summary.json:41`).

I rechecked the evidence locally:

- `rtk shasum -a 256 openhens_baseline_results/provenance/openhens-refactor-92e942f-to-2afc14b.patch`
  returned `cf18b6bdd96cca78fc3e6ac24a68e1327edcbcda5d9c7bb55e05b0e92695dd33`,
  matching `openhens_baseline_results/provenance/openhens-refactor-92e942f-to-2afc14b.patch.sha256`.
- `rtk git -C /Users/ca107/Desktop/ahuora/OpenHENS apply --check --reverse /Users/ca107/Desktop/ahuora/OpenPinch/openhens_baseline_results/provenance/openhens-refactor-92e942f-to-2afc14b.patch`
  exited 0 against the clean OpenHENS checkout.
- `rtk jq -e .` succeeded for the baseline manifest, refactor branch summary,
  and both Tier 0 summary files.

The task's reproducibility Definition of Done item is now checked
(`docs/developer/openhens-integration-tasks/00-baseline-freeze-and-acceptance-matrix.md:370`),
and the supporting implementation notes include the patch, hash, and reverse
apply evidence
(`docs/developer/openhens-integration-tasks/00-baseline-freeze-and-acceptance-matrix.md:390`).

### Resolved: `.DS_Store` is documented as excluded from the HENS-00 slice

The root `.DS_Store` is still dirty in the worktree, but HENS-00 now documents
it as a pre-existing, unrelated tracked file excluded from the implementation
slice. The manifest names `.DS_Store` and `docs/developer/index.rst` as excluded
dirty tracked files
(`openhens_baseline_results/baseline_manifest.json:17`,
`openhens_baseline_results/baseline_manifest.json:18`), and the task notes say
the root `.DS_Store` was not reverted or staged by this slice
(`docs/developer/openhens-integration-tasks/00-baseline-freeze-and-acceptance-matrix.md:410`).

That resolves the HENS-00 blocker as long as those unrelated dirty tracked files
remain excluded from the final task commit or PR.

## Checks That Still Pass

- Scope stayed within documentation, test metadata, and baseline artifacts. I
  found no modified OpenPinch production source, tests, packaging metadata, or
  lockfile changes in `OpenPinch`, `tests`, `pyproject.toml`, or `uv.lock`.
- Four-stream remains the routine baseline, and Nine-stream remains final
  verification only
  (`docs/developer/openhens-integration-tasks/00-baseline-freeze-and-acceptance-matrix.md:37`,
  `docs/developer/openhens-integration-tasks/00-baseline-freeze-and-acceptance-matrix.md:80`).
- The manifest still carries the required source paths and hashes, dependency
  and solver metadata, artifact schema/version, generated artifact paths,
  expected metrics, tolerances, verification commands, and missing solver rerun
  path
  (`openhens_baseline_results/baseline_manifest.json:74`,
  `openhens_baseline_results/baseline_manifest.json:102`,
  `openhens_baseline_results/baseline_manifest.json:157`,
  `openhens_baseline_results/baseline_manifest.json:181`,
  `openhens_baseline_results/baseline_manifest.json:219`,
  `openhens_baseline_results/baseline_manifest.json:225`).
- Fixture policy, artifact ownership, adapter snapshot paths, network snapshot
  paths, order-invariance expectations, and the migration acceptance matrix are
  concrete enough for later tasks
  (`docs/developer/openhens-integration-tasks/00-baseline-freeze-and-acceptance-matrix.md:168`,
  `docs/developer/openhens-integration-tasks/00-baseline-freeze-and-acceptance-matrix.md:192`,
  `docs/developer/openhens-integration-tasks/00-baseline-freeze-and-acceptance-matrix.md:222`,
  `docs/developer/openhens-integration-tasks/00-baseline-freeze-and-acceptance-matrix.md:247`,
  `docs/developer/openhens-integration-tasks/00-baseline-freeze-and-acceptance-matrix.md:272`).

## Residual Risks

- Full OpenHENS solver rerun is still blocked locally because Couenne and IPOPT
  are missing from PATH; the task records the exact rerun command and blocker.
- The root `.DS_Store` and `docs/developer/index.rst` remain dirty in the
  worktree. They are documented as excluded from HENS-00, but they must stay out
  of the final HENS-00 commit or PR.
- I did not rerun OpenPinch tests or docs build during re-review because this was
  requested as read-only and those commands may write caches/build output.

## Verdict

HENS-00 is cleared. The prior provenance blocker is resolved with a checked-in
patch plus hash and verified reverse apply, and the unrelated `.DS_Store` dirty
file is now explicitly excluded from the HENS-00 implementation slice.
