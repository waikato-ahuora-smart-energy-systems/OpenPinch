# Automatic patch-version requirements

## Intent and decision

The user requested automatic version bumps and selected A: patch by default;
major/minor increases remain explicit reviewed choices. This supersedes the
manual-version requirement in `automatic-release-requirements.md` and the
older delivery requirements. Automatic release after successful validation
remains required. This is a moderate-complexity delivery change with high-risk
source-write and irreversible-publication boundaries; standard requirements
depth applies. No application or thermodynamic behavior changes are intended.

## Acceptance requirements

1. For a new release without an explicit reviewed major/minor increase,
   automation selects the next safe patch version, for example 0.6.10 to
   0.6.11. Do not overwrite an existing tag or published distribution.
2. Honor an explicit reviewed major/minor increase instead of adding an
   unintended patch increment. No PR-label inference is introduced.
3. Complete normal review and its required tests before generating the automatic
   patch-bump PR; merge the reviewed changes and version preparation before main
   release validation. Update `pyproject.toml`, `.bumpversion.toml` and `uv.lock`
   consistently in a recorded source commit. The generated bump PR targets
   `develop`; merge it before merging the existing reviewed `develop` to `main`
   PR. Existing branch protections, including any renewed approvals, still apply.
4. Publication must use the exact distributions built and validated from the
   bumped main commit. Reuse successful normal-review tests only with verified
   source equivalence apart from strictly allowlisted version-field changes and
   preparation metadata. Rebuild and verify the bumped distributions, metadata,
   installation and provenance. Missing or invalid reuse proof requires normal
   validation, not an unchecked skip. This supersedes the blanket requirement
   to rerun all tests after the bump.
5. Bind version allocation to a release-preparation identity. Duplicate
   events, workflow retries and interrupted-release recovery reuse that
   version/commit; they must not allocate successive patch versions.
6. Prevent bump-triggered recursive release preparation. Concurrent events
   must not allocate the same version or overwrite newer source changes.
7. Preserve branch protections and required reviews/checks. Do not assume
   direct bot writes to main are permitted or silently grant bypass rights.
   The execution plan must establish a protection-compatible write path and
   identify any required GitHub configuration as a separate activation step.
8. Retain exact source/artifact provenance, complete validation coverage through
   fresh checks or independently verified reusable review evidence,
   TestPyPI then PyPI verification, immutable tags, stable-release checks and
   verified manual recovery. The original bundle remains the recovery source.
9. Report the original change, selected version, bumped commit and resulting
   release identity. Failed preparation/validation must not publish.
10. Preserve existing 0.6.10 artifacts; this change does not implicitly repair
    its legacy publication or authorize deleting, replacing or moving anything.

## Verification and scope

Use explicit regressions and generated properties for monotonic patch
selection, explicit major/minor preservation, consistent metadata, duplicate
events, retries, races, stale source, loop prevention and exact post-bump
validation. Test external mutations through fake boundaries, never real
publishing. Preserve current quality thresholds and validation lanes.

In scope: version-preparation helpers, workflow orchestration, delivery tests
and documentation. Out of scope: automatic merge, label-driven versioning,
application changes, live publication, direct remote source writes during
local implementation and unapproved permission/protection changes.

User stories remain skipped by prior user direction. Existing architecture
and delivery artifacts are reused; code-generation planning must define the
new source-write boundary before implementation.

## Extension compliance

Property-Based Testing remains enabled. PBT-01 through PBT-10 are N/A to
this requirements-only stage; their design/implementation obligations are
retained, especially idempotent allocation and stateful retry/race sequences.
Security and Resiliency extensions remain disabled and skipped. The explicit
publication and source-write safeguards above remain mandatory.

## Progress and review

- [x] Record option A and finalize acceptance criteria.
- [x] Obtain requirements approval before workflow planning (`Approve`).
- [x] Record subsequent user amendment: review first and reuse proven tests on main.
- [x] Resolve bump-PR target: `develop`, then existing reviewed PR into `main` (A).
- [x] Approve the revised execution plan (`Approve`).

A) Approve & Continue to Workflow Planning.

B) Request Changes to these requirements.

X) Other (describe the requested direction).

[Answer]: A (user replied `Approve`).
