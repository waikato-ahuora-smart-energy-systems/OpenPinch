# Unit 3 NFR Design Clarification Questions

## Ambiguity 1: Optimizer-Level Parallel Simulation

Question 2 selected option A, which keeps each targeting call sequential and
isolated, and added: “However, the optimiser may run simulations in parallel.”
The note could describe a future-compatible allowance or require Unit 3 to add
parallel candidate evaluation now. Those choices have different evaluator,
cache, process, ordering, and cleanup designs.

## Clarification Question 1
How should optimizer-level parallel TESPy candidate evaluation be scoped?

A) Keep Unit 3 execution sequential now, but design the evaluator and exact
cache values so a future approved optimizer may distribute independent
candidates across isolated processes. No parallel scheduler is implemented in
this unit.

B) Implement process-parallel TESPy candidate evaluation in Unit 3 now. Each
worker owns a TESPy evaluator session and bounded local cache; the parent
deduplicates exact requests, assigns stable ordinals, and merges results in
canonical order independently of completion order.

C) Permit concurrent callbacks within the current process and require one
shared TESPy evaluator and cache to support multi-threaded access.

X) Other (please describe after [Answer]: tag below)

[Answer]: A

## Content Validation

This clarification file contains no Mermaid diagram, ASCII diagram, executable
code block, JSON, or YAML. Markdown headings, choices, blank-line separation,
special characters, and the answer tag were checked before creation.
