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
