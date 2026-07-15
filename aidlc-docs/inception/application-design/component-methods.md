# Component Methods

- `Stream(..., segments=...) -> Stream`: construct a parent with validated ordered children.
- `Stream.from_temperature_heat_profile(...) -> Stream`: linearize and normalize profile data.
- `Stream.replace_segments(segments) -> None`: atomically replace a complete profile.
- `Stream.update_segment(index, **changes) -> None`: atomically update one child and revalidate the profile.
- `StreamCollection.segment_numeric_view(idx=None) -> StreamCollectionNumericView`: return expanded thermal rows with parent metadata.
- `StreamCollection.to_dict(idx=None, expand_segments=False) -> dict`: choose parent or expanded reporting.
- `problem_to_solver_arrays(...) -> PreparedSolverArrays`: add padded segment-profile tensors without changing parent axes.
- `partition_exchanger_duty_by_segments(...) -> tuple[HeatExchangerSegmentAreaContribution, ...]`: form ordered duty-aligned slices.
