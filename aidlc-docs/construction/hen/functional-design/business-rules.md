# Business Rules

- Matches, stages, and exchanger counts are parent based.
- Parent heat balance advances a cumulative heat coordinate.
- Temperature is obtained from the ordered piecewise profile, never aggregate CP.
- Pinch clipping may split a segment but cannot change parent identity or continuity.
- Exchanger duty is partitioned in thermal order at every hot or cold segment boundary.
- The topology-search objective uses the smooth Chen LMTD area surrogate.
- Reported, costed, and verified exchanger area is the sum of exact slice areas;
  multiperiod design area is the maximum period total.
- Extraction and feasibility checks reconcile all nested duties and areas.
