# Root Workflow Export Requirements

## Intent

Expose the two high-level workflow classes from the package root so user code
imports them with `from OpenPinch import PinchProblem, PinchWorkspace`.

## Requirements

- `OpenPinch.PinchProblem` shall be the concrete class owned by
  `OpenPinch.application.problem`.
- `OpenPinch.PinchWorkspace` shall be the concrete class owned by
  `OpenPinch.application.workspace`.
- The package root shall export exactly these two workflow classes through
  `__all__`; schemas, enums, services, and private records remain unexported.
- Root import shall remain usable without optional solver, dashboard, plotting,
  notebook, or heat-pump dependencies.
- User-facing documentation and packaged notebooks shall use the root import.
- Concrete owner modules remain the implementation owners and may continue to
  be used internally.

## Compatibility

No compatibility facade or legacy migration is required. This is the selected
public import contract going forward.
