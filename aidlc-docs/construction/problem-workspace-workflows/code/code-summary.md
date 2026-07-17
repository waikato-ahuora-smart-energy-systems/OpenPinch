# Unit 3 Code Generation Summary

The component surface is now `problem.components.add_process_mvr()` with a
read-only inventory. HEN design uses descriptive method names, ephemeral named
configuration overrides, fixed prerequisites, explicit multiperiod entry, and
an application view providing one-based `top`, `network`, selected-network
metrics, and lazy `grid` behavior while the underlying Pydantic result remains
serializable.

`PinchWorkspace` now presents cases only: unsolved `scenario()`, ordered
`cases()` batches with isolated per-case outcomes, active target/design/
component/plot/config forwarding, explicit comparisons, and canonical bundle
persistence. Variant aliases, workflow-string dispatch, redundant constructors,
and copy/get-case compatibility methods are absent. Summary/report/export
period modes are two booleans, and plots are observational through `data`,
`catalog`, named plot methods, and callable-selected exports.

The focused checkpoint passed 232 tests, including generated case-order and
non-mutation coverage; Ruff lint and format checks pass across 34 changed files.
Partial PBT rules are compliant. Security and Resiliency are disabled and N/A;
infrastructure design is N/A because no deployment resource changed.
