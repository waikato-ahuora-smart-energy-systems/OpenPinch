Documentation Conventions
=========================

The documentation site is meant to function as both a user manual and a public
API reference. That only works if documentation changes follow consistent
conventions.

Content Rules
-------------

- Document public surfaces by user intent first, then by module structure.
- Keep a clear separation between ``Stable``, ``Advanced``, and
  ``Experimental / partial`` capabilities.
- Explain the thermodynamic decision question before explaining the API call.
- Prefer one practical workflow example per page over long enumerations of
  loosely related features.
- When a new public method or CLI command is added, update both the curated
  narrative page and the exhaustive reference appendix.

Page Categories
---------------

``Overview``
   What the package does, when to use it, and how to choose a workflow.

``Fundamentals``
   Thermodynamic and architectural grounding in package terms.

``Guides``
   Task-oriented walkthroughs that answer one practical question.

``API``
   Curated public reference first, exhaustive module appendix second.

``Examples``
   Packaged notebooks, sample cases, and decision-workflow mapping.

Writing Style
-------------

- Use the same terminology as the code and the result tables.
- Write core OpenPinch concepts in spaced title case in prose, for example
  ``Total Site``, ``Total Process``, ``Heat Pump``, ``Problem Table``, and
  ``Composite Curve``. Reserve hyphenated forms for literal file names, page
  paths, CLI commands, graph IDs, and other API values.
- Avoid marketing language and generic claims about optimization.
- State package boundaries explicitly when a feature is partial or
  intentionally lower level.
- Link examples, guides, and API pages together so readers can move from
  concept to execution without guessing.

Docs Maintenance Expectations
-----------------------------

- Public API changes should ship with docs updates in the same change.
- New packaged notebooks or sample cases should be added to the examples pages.
- If a surface is intentionally experimental, keep it out of the stable guides
  and mark it clearly in the support matrix.
