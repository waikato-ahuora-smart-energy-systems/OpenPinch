# Integration Test Instructions: GitHub CI HEN Solver Isolation

## Purpose

Verify the interaction between the public design accessor, temporary HEN
configuration boundary, synthesis service, and fake workflow executor without
requiring external solver binaries.

## Scenario

1. Load the converted four-stream public example problem.
2. Reject a non-dictionary runtime-options object at the internal service
   boundary.
3. Reject HEN configuration supplied through the internal runtime-options
   boundary.
4. Invoke the public design accessor with call-local derivative thresholds.
5. Produce a synthesis result and manifest through `FakeSynthesisExecutor`.
6. Confirm stored derivative thresholds are restored after the call.

## Execution

```bash
/opt/homebrew/bin/uv run pytest -q \
  tests/analysis/heat_exchanger_networks/test_design_workflow.py::test_design_options_are_validated_at_their_owner_boundary
```

## Expected Result

- The single integration scenario passes.
- The design accessor and service retain their existing public and internal
  contracts.
- No external process, network service, database, Couenne, or IPOPT setup is
  required.

## Cleanup

No cleanup is required. Pytest's `monkeypatch` fixture restores every patched
executor reference at test teardown.
