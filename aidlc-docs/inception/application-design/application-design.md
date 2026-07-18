# Application Design Summary

The package-wide architecture modernization supersedes the earlier
stream-focused package-placement detail where ownership differs. Its current
design record is
`package-architecture-modernization-design.md` in this directory.

The design introduces an ordered child profile within the existing `Stream` aggregate. Collection and zone boundaries remain parent based. A shared segment projection prevents each numerical service from inventing its own flattening rules. HEN synthesis receives parent axes and segment tensors, preserving topology while replacing constant-CP equations with cumulative heat-coordinate relations. Reporting remains parent first and nests segment detail explicitly.

The approved package-usability refactor is documented separately in
`package-usability-refactor-design.md`. It preserves the two-class root facade,
adds descriptive target/design/workspace accessors, separates execution from
observation, and makes the tutorial/RTD manifest an enforced public-contract
consumer without overwriting the segmented-stream design above.

## Repository Issue Remediation Design

The remediation retains all existing public component boundaries. Workspace
case identifiers gain one shared strict validator used by runtime and bundle
entry points, with a second containment check at batch export. `problem_data`
becomes a detached observation boundary; explicit application methods remain the
only supported mutation paths. Reporting uses exclusive workbook reservation,
and the comparison tool uses one scoped import context that verifies every
OpenHENS module against the requested checkout before injecting its factory into
execution. Current documentation is corrected through a scoped drift guard that
does not rewrite historical records. No new runtime dependency or root export is
introduced.
