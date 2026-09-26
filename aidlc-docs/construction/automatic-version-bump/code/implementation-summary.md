# Automatic patch-version preparation implementation

Implemented strict inert-blob version transformation, deterministic create-only
preparation branches/PRs, review/run reconciliation, protected two-merge flow,
version-baseline gate, and independently verified review evidence reuse.

New scripts: release_preparation.py and review_evidence.py. Existing ci_policy.py
and release_manifest.py retain legacy recovery while adding delivery-v2 proof.
New notification/coordinator workflows use trusted main code for writes and
read-only review notification; shared validation/PR/main/develop/release callers
carry the required read-only evidence permissions. No new credentials or bypass.

Implementation refinements to Functional Design:

- Preparation record stores the tested merge commit as well as its tree, allowing
  reliable fetch and parent verification. Run retries do not change allocation
  identity; existing records are not rewritten when new evidence appears.
- Reuse accepts concrete full original review evidence, not recursive opaque
  reuse chains. The former pre-preparation develop aggregate shortcut is disabled.
  A missing proof reruns normal validation; no arbitrary reuse flag authorizes
  publisher success. Updated prepared PR/develop/main runs can reuse directly.
- General-test reuse still runs the complete packaging selection and the two
  identified version/config-sensitive non-packaging files fresh. Original review
  retains the 95 percent coverage gate. Docs/install checks always execute.
- Record identity plus original Git parents/metadata proves preparation; exact
  tree transformation is additionally required for test reuse. Later reviewed
  changes can keep the allocated version but must rerun normal validation.
- Manifest schema 2 embeds the verified preparation reference and binds it to
  exact artifact bytes; no extra release asset is introduced. Legacy schema 1
  remains all-jobs-success only. The publisher revalidates original review proof.

PBT compliance: PBT-01 properties documented in Functional Design; PBT-02 codec
and version round trips; PBT-03 metadata/identity/proof invariants; PBT-04 repeated
real coordinator calls; PBT-05 independent tuple/set oracles; PBT-06 generated
retry/interruption/closure sequences against real Git and fake API state;
PBT-07 bounded structured generators; PBT-08 seed 20260715 with shrinking;
PBT-09 existing Hypothesis; PBT-10 explicit regression cases alongside properties.
Security and Resiliency extensions remain disabled.

No hosted activation, push, PR creation, merge, release or index upload was run.
The repository version remains 0.6.10; legacy release recovery is not attempted.
