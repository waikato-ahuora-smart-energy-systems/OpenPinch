# Utility Placement Hierarchy API Clarification Questions

The hierarchy/profile answers are consistent, but two utility-count details
need clarification. Please place a letter after each `[Answer]:` tag.

## Clarification Question 1
Question 5 selected count override while the earlier requirement says existing
utilities allow counts to be inferred. Which complete behavior is intended?

A) If both counts are omitted, use the existing utilities as typed placement
templates and infer their counts. If either count is supplied, discard the
existing utilities for placement and generate a new generic template set from
the supplied counts.

B) If both counts are omitted, inspect existing utilities only to infer counts,
then generate generic templates; existing utility identities and attributes are
not retained as placement templates.

C) Existing utilities always remain the placement templates. Omitted counts are
inferred, while supplied counts must match them rather than override them.

D) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Clarification Question 2
Your Question 7 note says each level can be either a hot or cold utility. What
does `isothermal=2, sensible=2` mean when generic levels are requested?

A) Two isothermal plus two sensible levels in total across both hot and cold
sides, producing four utility levels altogether.

B) Two isothermal plus two sensible levels on each side, producing eight
utility levels altogether, as the current implementation does.

C) Other (please describe after the [Answer]: tag below)

[Answer]: B. The temperatures for the hot and cold utilities should be reversed. 4 levels should infer 8 utility entries.

## Clarification Question 3
If counts are totals across both sides, how should generic levels receive their
hot or cold role?

A) Split each kind as evenly as possible between hot and cold, requiring at
least one level on each side; reject an odd count because it cannot split
evenly.

B) Let the optimizer choose each level's hot or cold role, while requiring the
final set to cover both heating and cooling demand.

C) Require separate hot and cold counts from the caller instead of one total
count per kind.

D) This question is not applicable because counts are per side.

E) Other (please describe after the [Answer]: tag below)

[Answer]: The previous answers should resolve this question.
