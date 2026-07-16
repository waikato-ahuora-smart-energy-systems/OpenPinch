# Domain Entities

- `StreamSegment`: a `Stream` subclass with a stable segment index/name and optional owning parent.
- `Stream`: aggregate root containing zero or more ordered segments.
- `TemperatureHeatPointSchema`: temperature and cumulative heat coordinate.
- `TemperatureHeatProfileSchema`: ordered points and linearization tolerance.
- `StreamSegmentSchema`: one explicit linear thermal interval.
- Expanded numeric row: segment values plus parent key/index and segment index.
