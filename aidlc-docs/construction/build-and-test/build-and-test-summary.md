# Build and Test Summary

## Build Status

- Build tool: `uv` and `scripts/build_dist.py`.
- Status: success.
- Artifacts: OpenPinch 0.5.2 wheel and source distribution.
- Installed-artifact smoke: success outside the source checkout, including the
  eighteen packaged notebooks and the packaged Process MVR study input.
- Warning-as-error Sphinx HTML build: success.

## Test Execution Summary

### Unit, Property, and Integration Tests

- Complete non-solver profile: 2,084 passed, 3 optional-profile tests skipped,
  and 4 external-solver tests deselected.
- Final usability, notebook, documentation, architecture, workspace, component,
  and operation-manifest checkpoint: 108 passed and 3 opt-in profile selectors
  skipped.
- Notebook execution profiles: 10 base notebooks passed in routine pytest; all
  4 slow-HPR notebooks passed in 204.74 seconds; all 3 HEN solver notebooks
  passed in 160.68 seconds; the guarded interactive notebook passed in 7.78
  seconds.
- Internal method-selector contract checkpoint: 110 passed.
- Failed tests after corrections: 0.

### Static and Contract Gates

- Ruff lint: pass.
- Ruff format: pass across 458 files.
- Patch whitespace validation: pass.
- RTD operation coverage: 186 of 186 live operations mapped, including
  constructors, returned Process MVR behavior, and ordered case batches.
- Packaged tutorials: 18 of 18 structurally valid and mapped.
- Tutorial narrative: every notebook contains a study question, staged method
  guidance, interpretation guidance, and an adaptation section; the minimum is
  six Markdown cells per notebook.
- Stale public symbols: absent except explicit negative assertions.
- Offline warning-as-error Sphinx build: pass without network inventories.
- Brayton runtime status: explicitly documented as unsupported pending solver
  repair; tutorial 09 demonstrates the guarded comparison-study pattern.

### Performance and Security

- Performance load/stress testing: N/A; no performance requirement or service
  deployment changed.
- Security testing: N/A; the security extension is disabled and no security
  boundary changed.
- External binary solver tests remain outside the non-solver acceptance gate.
  The packaged HEN tutorial profile itself executed all three notebooks using
  the installed synthesis environment.

## Overall Status

- Build: success.
- Tests: pass.
- Documentation: pass.
- Distribution: pass.
- Operations: N/A because no deployment work was requested.
- Task status: complete and ready for review.
