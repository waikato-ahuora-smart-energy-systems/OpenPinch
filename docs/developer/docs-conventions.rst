Documentation Conventions
=========================

The documentation site is both a user manual for the package-root workflows
and a contributor reference for unsupported owners. That distinction only
works if documentation changes follow consistent conventions.

Content Rules
-------------

- Document the protected main contract by user intent first.
- Label every deep-owner workflow ``Unsupported internal``; never imply that
  tested repository functionality is compatibility-protected.
- Explain the thermodynamic decision question before explaining the API call.
- Keep each guide on the standard structure: purpose, prerequisites, sample
  case or asset, runnable workflow, expected output, interpretation, and next
  steps.
- Prefer one practical workflow example per guide over long enumerations of
  loosely related features.
- When the main contract or a CLI command changes, update both the curated
  narrative page and the relevant reference page.
- Do not present removed solver, graph-export, or validation command surfaces
  as supported CLI workflows.

Page Categories
---------------

``Overview``
   What the package does, when to use it, and how to choose a workflow.

``Fundamentals``
   Thermodynamic and architectural grounding in package terms.

``Guides``
   Task-oriented walkthroughs that answer one practical question.

``API``
   The protected main contract first, unsupported module appendix second.

``Examples``
   Packaged notebooks, sample cases, and decision-workflow mapping.

Writing Style
-------------

- Use the same terminology as the code and the result tables.
- Write core OpenPinch concepts in spaced title case in prose, for example
  ``Total Site``, ``Indirect``, ``Heat Pump``, ``Problem Table``, and
  ``Composite Curve``. Reserve hyphenated forms for literal file names, page
  paths, CLI commands, graph IDs, and other API values.
- Avoid marketing language and generic claims about optimization.
- State package boundaries explicitly when a feature is partial or
  intentionally lower level.
- Link examples, guides, and API pages together so readers can move from
  concept to execution without guessing.

Docs Maintenance Expectations
-----------------------------

- Main-contract changes should ship with docs updates in the same change.
- Changes to the protected contract must update both a task-oriented page and
  the curated API page for that contract.
- New packaged notebooks or sample cases should be added to the examples pages.
- New packaged notebooks or sample cases should update docs consistency tests
  that compare examples docs with ``OpenPinch.resources``.
- Keep unsupported owner examples clearly labelled in guides, notebooks, and
  the support matrix.
- Legacy pages under ``docs/user-guide`` and moved ``docs/reference`` entry
  points should remain as orphan transition pages unless a deliberate redirect
  strategy replaces them.
