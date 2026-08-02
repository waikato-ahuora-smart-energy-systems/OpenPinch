# Notebook Improvement Requirements Questions

Please answer every question by entering one letter after its `[Answer]:` tag.
If an option does not fit, choose the final `Other` option and add a short
description after the letter.

The repository currently has generated edits in all 18 packaged tutorial
notebooks and an untracked `openpinch-workspace.json` file. No existing notebook
change will be overwritten while these answers are pending.

## Question 1
Which notebooks should this improvement cover?

A) All 18 packaged tutorial notebooks

B) A specific subset of the 18 packaged tutorial notebooks (list them after the `[Answer]:` tag)

C) One or more new packaged tutorial notebooks

D) Only `examples/debug_support_notebook.ipynb`

X) Other (please describe after `[Answer]:` tag below)

[Answer]: A

## Question 2
What is the primary improvement goal?

A) Improve instructional clarity, explanations, study questions, and adaptation guidance

B) Improve execution reliability and ensure every example uses the current public API correctly

C) Expand workflow and API coverage with additional realistic process-engineering examples

D) Improve visual presentation, plots, reports, and retained outputs

E) Apply a comprehensive improvement combining instructional quality, reliability, coverage, and presentation

X) Other (please describe after `[Answer]:` tag below)

[Answer]: D. Many of the tutorials end with limited visuals, printed graphs or outputs

## Question 3
How should the current uncommitted notebook edits and generated `openpinch-workspace.json` be treated?

A) Treat them as the intended baseline, preserve them, and build the improvement on top

B) Review and diagnose them first, then present findings before changing notebook content

C) Treat generated notebook outputs and the workspace file as disposable; regenerate from the canonical generator after requirements approval

X) Other (please describe after `[Answer]:` tag below)

[Answer]: X - They are gone.

## Question 4
What should be the canonical source of notebook content after this work?

A) Update `scripts/generate_tutorial_notebooks.py` and regenerate the packaged notebooks from it

B) Edit the selected `.ipynb` files directly and leave the generator unchanged

C) Keep the generator authoritative for structure, but allow intentional notebook-specific output and metadata updates after generation

X) Other (please describe after `[Answer]:` tag below)

[Answer]: A

## Question 5
What execution verification is required for the improved notebooks?

A) Execute every applicable profile, including slow heat-pump and external HEN solver profiles when available

B) Execute all base and non-external profiles; validate external-solver notebooks structurally without requiring solver execution

C) Execute only the specifically changed notebooks under their declared profiles

D) Perform structural and static validation only, without executing notebooks

X) Other (please describe after `[Answer]:` tag below)

[Answer]: A

## Question 6
What should define completion for the learning experience?

A) Each notebook has clear prerequisites, learning outcomes, executable steps, interpretation, study questions, and adaptation guidance

B) Prioritize concise runnable examples; retain only minimal explanatory prose

C) Prioritize comprehensive process-engineering narrative even if notebooks become longer

X) Other (please describe after `[Answer]:` tag below)

[Answer]: C - the focus is on demonstrating with no user input

## Question 7
Should security extension rules be enforced for this project?

A) Yes — enforce all SECURITY rules as blocking constraints (recommended for production-grade applications)

B) No — skip all SECURITY rules (suitable for PoCs, prototypes, and experimental projects)

X) Other (please describe after `[Answer]:` tag below)

[Answer]: B

## Question 8
Should the resiliency baseline be applied to this project?

The resiliency baseline provides directional, design-time guidance for fault
tolerance, availability, observability, recoverability, disaster recovery, and
continuous improvement. It is a starting point, not production-readiness
certification or a substitute for a formal AWS Well-Architected Review.

A) Yes — apply the resiliency baseline as directional best practices and design-time guidance

B) No — skip the resiliency baseline (suitable for local tutorials and experimental projects)

X) Other (please describe after `[Answer]:` tag below)

[Answer]: B

## Question 9
Should property-based testing rules be enforced for this project?

A) Yes — enforce all PBT rules as blocking constraints (recommended for business logic, transformations, serialization, or stateful components)

B) Partial — enforce PBT rules only for pure functions and serialization round-trips

C) No — skip all PBT rules (suitable when notebook prose and presentation are the only changes)

X) Other (please describe after `[Answer]:` tag below)

[Answer]: B
