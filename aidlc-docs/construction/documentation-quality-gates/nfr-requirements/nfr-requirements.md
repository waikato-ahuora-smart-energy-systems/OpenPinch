# NFR Requirements

- Manifest and notebook generation must be deterministic and produce stable
  ordering.
- Live public inventory drift must fail tests before undocumented operations
  can ship.
- Base tutorials must execute from clean temporary directories in routine CI.
- Optional slow HPR and solver profiles must be explicitly identifiable and
  runnable in environments that provide their dependencies.
- Sphinx must build with warnings treated as errors.
- Wheel and source archives must contain the canonical notebook inventory.
- Quality gates must avoid network access and hidden application execution.
