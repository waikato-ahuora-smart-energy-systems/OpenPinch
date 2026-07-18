# NFR Requirements

- Preserve case and period insertion order in serial and threaded execution.
- Keep invocation overrides isolated from persistent problem configuration.
- Importing the base package must not load plotting or solver extras.
- Missing prerequisites, invalid ranks, invalid case names, and unsupported
  arguments must raise actionable deterministic exceptions.
- Public methods must be introspectable and avoid OpenPinch-owned closed-string
  selectors.
- Focused and property-generated tests must prove non-mutation and ordering.

Security, service availability, infrastructure, and horizontal scaling are N/A:
this unit is an in-process engineering library with no new trust boundary.
