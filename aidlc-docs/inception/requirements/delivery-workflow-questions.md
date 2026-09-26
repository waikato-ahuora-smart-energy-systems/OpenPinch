# Delivery workflow policy

## Question 1: Release initiation

Should every merge to main continue to publish a package release?

A) Explicit releases (recommended): keep develop/main, validate normal merges,
and use a deliberate release-preparation PR or action to select and commit the
version before final validation and publishing. Normal merges do not publish.

B) Release every main merge: preserve automatic publication, but prepare the
version on develop before validation and require the complete PR gate.

X) Other (please describe after the answer tag).

[Answer]: A

## Existing extension decisions

These carry forward from the recorded project configuration; no new answer is
needed unless the user wants to change them.

- Property-Based Testing: enabled in full.
- Security Baseline: disabled.
- Resiliency Baseline: disabled.

The proposed scope and evidence are in `delivery-workflow-investigation.md`.
No remote publication or settings change is included in this policy choice.
