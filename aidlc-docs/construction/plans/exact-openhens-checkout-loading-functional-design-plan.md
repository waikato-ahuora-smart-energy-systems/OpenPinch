# Exact OpenHENS Checkout Loading Functional Design Plan

## Context Resolution

The approved requirements, Application Design, and Unit 2 definition resolve
all functional choices. No unanswered `[Answer]:` tags are required.

- **Business logic modeling**: temporarily isolate the `openhens` import
  namespace, load from the requested checkout, validate origins and
  capabilities, inject the verified factory, and restore interpreter state.
- **Domain model**: the requested checkout path, required module set, verified
  factory, and interpreter import-state snapshot are the only unit concepts.
- **Business rules**: no ambient-cache fallback, no compatibility patching, no
  execution through an unverified factory, and failure before output creation.
- **Data flow**: checkout path to scoped imports to validation to factory-backed
  source execution, followed by unconditional restoration.
- **Integration points**: Python import machinery and the existing comparison
  script only; no package API, solver, network, or persistence change.
- **Error handling**: actionable missing-directory, import, foreign-origin, and
  missing-capability errors with restoration on every exit path.
- **Business scenarios**: cached foreign modules, duplicate path entries,
  requested imports, partial import failure, invalid capability, successful
  execution, and execution failure.
- **Frontend components**: N/A; this is a repository comparison utility.

## Generation Checklist

- [x] Analyze Unit 2 responsibilities, FR-3, assigned NFRs, and exclusions.
- [x] Define the scoped exact-checkout import lifecycle.
- [x] Define module-origin and callable-capability validation.
- [x] Define verified-factory injection into source execution.
- [x] Define success/failure restoration of `sys.path` and `sys.modules`.
- [x] Define failure-before-output behavior.
- [x] Generate `business-logic-model.md`.
- [x] Generate `business-rules.md`.
- [x] Generate `domain-entities.md`.
- [x] Validate completeness against FR-3 and acceptance criteria 1 and 4.
- [x] Confirm no frontend artifact is applicable.
