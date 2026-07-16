# Stream Refactor Scope Questions

Please record one letter after each `[Answer]:` tag.

## Question 1
What depth should the `stream.py` refactor use?

A) Extract value/period helpers, derived thermodynamic calculations, and segmented-profile calculations into three private modules while retaining orchestration in `Stream` (recommended).

B) Extract only segmented-profile calculations into one private helper as the smallest change.

C) Move most implementation, including `StreamSegment`, into private implementation modules and keep `stream.py` primarily as a facade.

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 2
What compatibility boundary should the refactor follow?

A) Preserve all behavior, exceptions, public imports, class module identities, serialization, pickle compatibility, and property/method signatures (recommended).

B) Preserve public imports and serialization but allow documented internal behavior cleanup and exception-message changes.

X) Other (please describe after the [Answer]: tag below)

[Answer]: A

## Question 3
How should implementation be sequenced?

A) Use one approved plan but extract and verify one responsibility at a time, finishing with the full proportional suite (recommended).

B) Perform the complete file decomposition first and test only after all modules are extracted.

X) Other (please describe after the [Answer]: tag below)

[Answer]: A
