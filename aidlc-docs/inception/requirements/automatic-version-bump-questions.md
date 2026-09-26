# Automatic version bumps

The user has replaced the earlier manual-version policy with automatic version
bumps. Existing configuration can update `pyproject.toml`, `uv.lock` and the
bump configuration together. The bumped source commit must pass full validation
before its exact packages are published; do not bump an already-tested artifact.

## Question 1: Bump selection

Which rule should automation use to choose the next version?

A) Patch by default (recommended): each new release advances the patch version,
for example 0.6.10 to 0.6.11. Major/minor increases remain explicit reviewed
choices. Rerunning an interrupted release must not allocate another version.

B) PR-label driven: use an explicit major or minor label when present, otherwise
patch. Conflicting labels block release preparation.

X) Other (describe the desired version rule).

[Answer]: A

## Preserved safeguards

- Commit the selected version before full release-candidate validation/build.
- Retain immutable source/artifact identity and verified manual recovery.
- Prevent duplicate bumps and recursive CI/release triggers.
- Preserve branch protections; do not assume a bot may bypass them.
- No existing tag or published package is overwritten.
- No live version bump, push or package publication is performed during planning.
- Existing PBT enabled; Security/Resiliency extensions remain disabled.

## Progress

- [x] Inspect existing version configuration and record changed user intent.
- [x] Resolve bump selection and amend requirements.
- [ ] Approve requirements and prepare the implementation plan.
