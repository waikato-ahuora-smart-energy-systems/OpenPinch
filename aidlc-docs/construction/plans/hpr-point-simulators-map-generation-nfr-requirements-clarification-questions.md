# Unit 2 NFR Requirements Clarification Questions

The response to original Question 5 states, “REFPROP should be rejected; only
CoolProp (>=8) should be used.” The intent to reject REFPROP is clear, but the
response combines a package-version decision with a property-backend decision.
Both affect compatibility and dependency metadata and therefore require exact
choices.

Please fill every `[Answer]:` tag with one listed letter. Choose `X` and add a
description when none of the listed options matches the intended requirement.

## Clarification Question 1
Should this work raise OpenPinch's required CoolProp package baseline from its
current unbounded declaration and locked 7.2.0 environment to version 8 or
newer?

A) Yes. Require `CoolProp>=8` for the complete package, update the lockfile, and
run all existing thermodynamic regressions against the upgraded dependency.

B) No. Keep the current CoolProp dependency policy; “>=8” was descriptive, not
a requested OpenPinch dependency change.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## Clarification Question 2
After rejecting REFPROP, which CoolProp property backends may a working-fluid
specification select?

A) HEOS only for HPR map generation. Reject REFPROP and every other explicit
backend in the Unit 2 context.

B) Any installed CoolProp-native backend that can evaluate the required cycle
states, such as HEOS, SRK, or PR; reject REFPROP specifically because it is an
external licensed provider.

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: B

## Content Validation

This clarification file contains no Mermaid diagram, ASCII diagram, executable
code block, embedded JSON, or YAML. Markdown headings, option spacing, answer
tags, dependency notation, and punctuation were validated before creation.
