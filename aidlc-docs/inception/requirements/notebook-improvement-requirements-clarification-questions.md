# Notebook Improvement Requirements Clarification Questions

The initial answers are complete, but two presentation decisions remain
ambiguous. They affect the packaged notebook contract and the implementation
approach, so they must be resolved before the requirements document is created.

Please answer each question by entering one letter after its `[Answer]:` tag.

## Ambiguity 1: Retained outputs versus source-only notebooks

The improvement goal mentions limited visuals and outputs. The current packaged
notebook contract deliberately stores no execution counts or cell outputs.

### Clarification Question 1
Should generated notebooks remain source-only in Git?

A) Yes — keep stored outputs empty, but make execution display useful inline plots, tables, summaries, and result objects

B) No — commit executed cell outputs and execution counts for all 18 notebooks

C) Commit representative outputs only in selected notebooks and keep the others source-only

X) Other (please describe after `[Answer]:` tag below)

[Answer]: A

## Ambiguity 2: Preferred demonstration pattern

The tutorials can demonstrate results inline, through exported artifacts, or
both. This choice affects notebook length, temporary-file handling, and visual
verification.

### Clarification Question 2
How should each tutorial conclude its demonstration?

A) Show the most relevant inline plot or table, followed by concise engineering interpretation

B) Generate reviewable files such as HTML, Excel, or image exports and show their paths

C) Combine inline plots or tables with exported artifacts wherever the workflow supports both

X) Other (please describe after `[Answer]:` tag below)

[Answer]: A
