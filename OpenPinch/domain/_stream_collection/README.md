# Stream Collection Helpers

## Overview
Private helpers behind `OpenPinch.domain.stream_collection.StreamCollection`: filtered subset construction, the dense numeric snapshot used by analysis kernels, report serialisation and sort and pickle helpers. Each function takes the collection (or its streams) as an argument and reads its private state.

## Interface
- `build_stream_subset` (`filters.py`): new collection of existing streams filtered by type, process or utility role, with optional inversion of utility streams and sort override.
- `StreamCollectionNumericView` (`numeric_view.py`): frozen dataclass of NumPy arrays (`t_min`, `t_max`, shifted temperatures, `cp`, `rcp`, `heat_flow`, `dt_cont`, `is_hot`, `is_cold`, `active`, plus parent and segment index metadata).
- `build_numeric_view`, `build_segment_numeric_view` (`numeric_view.py`): build the view per parent stream, or expanded to segments.
- `value_at_idx` (`numeric_view.py`): magnitude of a scalar or multiperiod value at a period index, NaN for `None`.
- `collection_to_dict` (`serialization.py`): column-oriented rows in reporting order, optionally expanded to segments.
- `_sort_by_attr`, `_sort_by_attrs`, `_stream_attr_value`, `_is_picklable` (`sorting.py`): the private functions the collection imports for sort keys, per-period attribute reads and the pickle probe.

## Invariants
- A subset shares its parent's period context and sort settings, holds the same stream objects (inverted utility streams are shallow copies that are inverted), and is flagged for re-sorting. [filters.py:build_stream_subset]
- When `invert_utility` is set, only utility streams of the opposite type to the requested one are returned. [filters.py:build_stream_subset]
- Numeric views are NaN where a value is `None`, and a scalar value is used for every period. [numeric_view.py:value_at_idx]
- Segment views list a stream's segments in order, or the stream itself with `segment_index` of -1. [numeric_view.py:build_segment_numeric_view]
- Report rows sort by category (hot stream, cold stream, hot utility, cold utility, other) and descending temperatures, with the same column order every call. [serialization.py:collection_to_dict]
- Period indexes are non-negative integers. [numeric_view.py:value_at_idx] [sorting.py:_stream_attr_value]

## Side Effects
- `build_stream_subset` calls `invert()` on copies of utility streams, not the originals. [filters.py:build_stream_subset]
- `_is_picklable` serialises its argument with `pickle.dumps` and discards the bytes. [sorting.py:_is_picklable]

## Verification
- Run: `uv run --no-sync pytest tests/domain/test_stream_collection.py`
