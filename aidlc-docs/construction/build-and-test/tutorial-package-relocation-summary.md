# Tutorial package relocation verification

Completed relocation of notebooks and sample_cases from OpenPinch/data to
OpenPinch/tutorials. All 32 original files remain byte-identical, including saved
notebook outputs and openpinch-workspace.json. Runtime maps and interchange
contracts retain their existing data packages. Resource helpers and sample-name
resolution retain their behavior.

Updated resource roots, both asset generators, affected tests, Ruff exclusions,
README and API documentation. Corrected the stale claim that packaged notebooks
have no saved execution outputs. Historical workflow records retain historical
paths.

## Verification

- 164 selected packaging and affected application/array-adapter tests passed;
  14 notebook execution cases were deselected. This includes strict Sphinx docs,
  resource copying, notebook validity and generator preservation/repeatability.
- Executed the first tutorial separately: 1 passed. Full numerical and optional
  notebook execution suites were not repeated for this package-layout change.
- Wheel and sdist built successfully. Rebuilt a wheel from the extracted sdist.
- Installed both wheels into separate temporary targets and verified imports
  resolve from those targets. All 32 file hashes match in both installations;
  all 19 notebooks and 10 sample cases copy correctly, and every sample loads
  by name. Installed artifact smoke checks passed with the TESPy profile,
  including runtime maps and contracts.
- The first core-only artifact smoke attempt rejected the development
  environment because TESPy is installed. Reran using the matching TESPy
  profile; both installations passed. No package defect or code change needed.
- Ruff check, formatting of all ten affected Python files and git diff whitespace
  checks passed. No obsolete tutorial paths remain in current code or docs.

Test commands and output are recorded in /tmp/openpinch-tutorial-relocation-tests.log,
/tmp/openpinch-tutorial-relocation-notebook.log,
/tmp/openpinch-tutorial-relocation-build.log and
/tmp/openpinch-tutorial-relocation-installed.log. Distribution artifacts are under
/tmp/openpinch-tutorial-relocation-dist.

## Extension applicability

PBT: compliant through existing generator repeatability/preservation and resource
copy properties, plus byte identity checks. Security and Resiliency: disabled in
the existing extension configuration, so skipped. Performance and infrastructure
work: not applicable to a package-layout refactor. No commit or push performed.
