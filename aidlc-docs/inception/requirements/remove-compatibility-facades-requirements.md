# Remove Compatibility Facades Requirements

## Status

Approved by the user's explicit clean-break implementation request.

## Intent

Remove modules whose sole purpose is preserving superseded synthesis-schema
import or pickle paths. Route all package exports, implementation imports,
tests, and documentation directly to concrete owner modules.

## Requirements

- Remove `lib.schemas.synthesis.methods`, `tasks`, and `results`.
- Make `lib.schemas.synthesis.__init__` package-only rather than a re-exporting
  compatibility barrel.
- Remove old barrel-qualified synthesis imports and pickle compatibility.
- Retain concrete `common`, `topology`, `method`, `task`, and `result` modules.
- Retain intentional public API barrels at `OpenPinch`, `OpenPinch.lib`, and
  `OpenPinch.lib.schemas`, but map their synthesis names directly to concrete
  owner modules.
- Preserve schema behavior, names, validation, JSON schemas, dumps, solver
  behavior, and result structures; only compatibility import/pickle paths break.

## Workflow Decisions

- Existing reverse-engineering and package-reorganization artifacts are reused.
- User stories, functional design, NFR design, infrastructure, and Operations
  are skipped because this is a narrow import-structure cleanup.
- Security and Resiliency remain disabled. Partial Property-Based Testing is
  N/A because no algorithmic or serialization behavior is added.
