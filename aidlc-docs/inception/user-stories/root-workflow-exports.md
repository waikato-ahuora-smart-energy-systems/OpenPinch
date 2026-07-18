# Root Workflow Export User Story

As an OpenPinch user, I want to import `PinchProblem` and `PinchWorkspace`
directly from `OpenPinch`, so that the primary workflow entry points are concise
and discoverable.

## Acceptance Criteria

- `from OpenPinch import PinchProblem, PinchWorkspace` succeeds.
- The imported objects are identical to their concrete application owners.
- No other public root exports are introduced.
- Root cold-import tests pass without optional dependencies.
- User guides and packaged notebooks demonstrate the root import.
