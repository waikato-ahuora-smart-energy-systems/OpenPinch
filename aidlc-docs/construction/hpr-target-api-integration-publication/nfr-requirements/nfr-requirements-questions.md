# Unit 3 NFR Requirements Questions

Please answer both questions to settle the remaining performance and evaluation
policy choices for TESPy-backed targeting.

## Question 1
How should repeated evaluation of an exactly identical TESPy optimizer candidate
be handled within one targeting call?

A) Use a bounded, call-local cache keyed by the complete normalized candidate
request. Reuse only immutable successful or candidate-local failure results;
never share entries across targeting calls or use approximate float matching.

B) Run a fresh TESPy design solve for every objective callback, including exact
duplicate candidates. No candidate results are cached.

C) Use tolerance-based candidate caching so nearby temperature and duty vectors
may share a TESPy result.

X) Other (please describe after [Answer]: tag below)

[Answer]: A

## Question 2
What performance gate should apply to environment-sensitive TESPy targeting?

A) Use deterministic structural gates plus a generous marked smoke timeout:
bound TESPy solve calls by unique candidate requests, prohibit growth across
repeated calls, and require one single-stage target plus map workflow to finish
within five minutes on the supported CI profile. Record elapsed time as trend
evidence rather than imposing a tight workstation-specific target.

B) Require each complete single-stage TESPy target plus map workflow to finish
within sixty seconds on every supported developer and CI environment.

C) Apply call-count and memory-growth checks only, with no wall-clock timeout or
trend evidence for real TESPy integration.

X) Other (please describe after [Answer]: tag below)

[Answer]: A

## Content Validation

This question file contains no Mermaid diagram, ASCII diagram, executable code
block, JSON, or YAML. Markdown headings, choices, blank-line separation,
special characters, units, and answer tags were checked before creation.
