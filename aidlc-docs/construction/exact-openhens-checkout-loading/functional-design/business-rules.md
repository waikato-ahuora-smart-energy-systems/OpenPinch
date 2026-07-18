# Business Rules

## Checkout Identity

- **OPENHENS-01**: The requested root must exist as a directory.
- **OPENHENS-02**: The requested root is first in the scoped import path,
  regardless of whether the same string appeared elsewhere in the original
  path.
- **OPENHENS-03**: Ambient `openhens` and `openhens.*` cache entries cannot
  satisfy requested imports.
- **OPENHENS-04**: Every required imported module must expose a source file
  beneath the resolved requested root. Missing or foreign origins fail closed.
- **OPENHENS-05**: No fallback to installed, cached, or partially matching
  OpenHENS packages is allowed.

## Capabilities and Execution

- **OPENHENS-06**: The fixed upstream capabilities remain callable:
  `OpenHENS`, `run_parallel_solutions`, and both `OrganiseArray` definitions.
- **OPENHENS-07**: Validation must not add, remove, or replace attributes on
  imported upstream modules.
- **OPENHENS-08**: Source execution receives the already verified `OpenHENS`
  callable explicitly and performs no ambient second import.
- **OPENHENS-09**: Solver parameters, model construction arguments, result
  ranking, and output formats remain unchanged.

## Restoration and Failure

- **OPENHENS-10**: Original `sys.path` order and multiplicity are restored on
  success and every failure.
- **OPENHENS-11**: Original cached OpenHENS module object identities are restored
  on success and every failure; modules created by the scope are removed.
- **OPENHENS-12**: Missing, unimportable, foreign, or unsupported checkouts raise
  actionable `RuntimeError` messages naming the requested checkout and cause.
- **OPENHENS-13**: Real-comparison preflight fails before output-directory
  creation.
