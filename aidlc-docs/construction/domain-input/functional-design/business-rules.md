# Business Rules

- Preserve supplied segment order; never sort.
- Every segment has positive duty and a consistent parent hot/cold direction.
- For every period, one segment target equals the next segment supply within thermal tolerance.
- Parent endpoints are the outer segment endpoints and parent duty is the sum of children.
- Profiles are authoritative and duplicated parent aggregates are assertions.
- Mutations validate a candidate copy before committing and roll back on failure.
- Flat schema rows are never grouped.
- Child revisions contribute to collection numeric-cache identity.
