"""Utility container for managing ordered sets of stream objects."""

from __future__ import annotations

import math
import warnings
from copy import copy, deepcopy
from functools import partial
from typing import Any, Callable, List, Tuple, Union

import numpy as np

from ._stream_collection.filters import build_stream_subset
from ._stream_collection.numeric_view import (
    StreamCollectionNumericView,
    build_numeric_view,
    build_segment_numeric_view,
    value_at_idx,
)
from ._stream_collection.serialization import collection_to_dict
from ._stream_collection.sorting import (
    _is_picklable,
    _sort_by_attr,
    _sort_by_attrs,
    _stream_attr_value,
)
from .enums import ST
from .stream import Stream


class StreamCollection:
    """A dynamic, ordered collection of streams.

    Key features include:

    - Add and remove streams by name.
    - Prevent overwriting existing streams by auto-renaming.
    - Configure sort keys as attributes or callables.
    - Iterate efficiently with lazy sorting.
    - Support ascending or descending ordering.
    """

    def __init__(self, streams: List["Stream"] | None = None):
        """Initialise an empty collection sorted by descending supply temperature."""
        self._streams: dict[str, object] = {}
        self._period_ids: dict[str, int] | None = {"0": 0}
        self._weights: np.ndarray | None = np.array([1.0])
        self._sort_spec: Tuple[str, Any] = ("attr", "t_supply")
        self._sort_key: Callable = partial(_sort_by_attr, "t_supply")
        self._sort_reverse: bool = True
        self._sorted_cache: List[object] = []
        self._needs_sort: bool = True
        self._numeric_cache: dict[
            tuple[str, int | None, tuple],
            StreamCollectionNumericView,
        ] = {}
        self._num_periods: int | None = 1
        if streams is not None:
            self.add_many(streams)

    @property
    def period_ids(self) -> dict[str, int] | None:
        """Return the canonical period identifiers for this collection."""
        return self._period_ids

    @property
    def weights(self) -> np.ndarray | None:
        """Return the canonical period weights for this collection."""
        return self._weights

    @property
    def num_periods(self) -> int | None:
        """Return the number of periods for this collection."""
        return self._num_periods

    def _rebuild_sort_key(self):
        mode, sort_detail = self._sort_spec
        if mode == "attr":
            self._sort_key = partial(_sort_by_attr, sort_detail)
        elif mode == "attrs":
            self._sort_key = partial(_sort_by_attrs, sort_detail)
        else:
            self._sort_key = sort_detail

    def add(
        self, stream: "Stream", key: str = None, prevent_overwrite: bool = True
    ) -> str:
        """Insert a stream, optionally renaming the key to avoid collisions."""
        self._validate_stream_period_context(stream)
        self._adopt_appropriate_period_context(stream, stream)
        base_name = new_name = stream.name
        if key is None:
            key = base_name

        original_key = key
        counter = 1
        while prevent_overwrite and key in self._streams:
            key = f"{original_key}_{counter}"
            new_name = f"{base_name}_{counter}"
            counter += 1
        stream.name = new_name
        self._streams[key] = stream

        stream.set_period_context(
            weights=self._weights,
            period_ids=self._period_ids,
            num_periods=self._num_periods,
        )
        self._needs_sort = True
        self._invalidate_numeric_cache()
        return key

    def add_many(
        self,
        streams: List["Stream"],
        keys=None,
        prevent_overwrite: bool = True,
    ):
        """Insert several streams, optionally using explicit keys for each stream."""
        if keys is None:
            for stream in streams:
                self.add(stream, prevent_overwrite=prevent_overwrite)
        else:
            if len(streams) != len(keys):
                raise ValueError("Length of streams and keys must match.")
            for stream, key in zip(streams, keys):
                self.add(stream, key, prevent_overwrite)

    def get_stream_by_name(self, name: str, approximate: bool = False) -> Stream:
        for stream in self:
            if (stream.name == name) or (approximate and name in stream.name):
                return stream
        warnings.warn(f"Stream '{name}' not found.")
        return None

    def get_stream_names(self) -> list:
        return [stream.name for stream in self._streams.values()]

    def remove(self, stream_name: str):
        """Remove a stream by name."""
        if stream_name in self._streams:
            del self._streams[stream_name]
            self._needs_sort = True
            self._invalidate_numeric_cache()
        else:
            warnings.warn(f"Stream '{stream_name}' not found.")

    def sum_stream_attribute(self, attr_name: str, idx: int | None = None) -> float:
        """Return the total of a specified attribute for streams in the collection."""
        if self._streams is None or len(self._streams) == 0:
            warnings.warn(
                f"Attempted to sum attribute '{attr_name}' "
                "on an empty stream collection."
            )
            return 0.0
        stream = next(iter(self._streams.values()))
        if hasattr(stream, attr_name):
            return sum(
                _stream_attr_value(stream, attr_name, idx)
                for stream in self._streams.values()
            )
        warnings.warn(f"Stream '{stream.name}' does not have attribute '{attr_name}'.")
        return 0.0

    def set_common_stream_attribute(
        self,
        attr_name: str,
        value: Any,
        *,
        idx: int | None = None,
    ):
        """Set a common attribute across all streams in the collection."""
        if self._streams is None or len(self._streams) == 0:
            warnings.warn(
                f"Attempted to set attribute '{attr_name}' "
                f"on an empty stream collection."
            )
            return 0.0
        for stream in self._streams.values():
            if not hasattr(stream, attr_name):
                warnings.warn(
                    f"Stream '{stream.name}' does not have attribute '{attr_name}'."
                )
                continue
            current_value = _stream_attr_value(stream, attr_name, idx)
            if current_value == value:
                continue
            if idx is None:
                setattr(stream, attr_name, value)
            else:
                stream.set_value_attr_at_idx(attr_name, value, idx=idx)
            self._invalidate_numeric_cache()
        return self

    def set_sort_key(self, key: Union[str, List[str], Callable], reverse: bool = False):
        """Set the sorting key. Supports attribute names or custom lambdas."""
        self._sort_reverse = reverse
        if isinstance(key, str):
            self._sort_spec = ("attr", key)
        elif isinstance(key, list):
            self._sort_spec = ("attrs", tuple(key))
        else:
            self._sort_spec = ("callable", key)
        self._rebuild_sort_key()
        self._needs_sort = True

    def copy(
        self,
        *,
        deep: bool = False,
    ) -> "StreamCollection":
        """Return a copy of the collection, optionally deep-copying streams."""
        return deepcopy(self) if deep else copy(self)

    def set_period_context(
        self,
        period_ids: dict[str, int] | list[str] | tuple[str, ...] | None,
        weights: np.ndarray | list[float] | tuple[float, ...] | None,
        num_periods: int | None = None,
    ) -> None:
        """Persist the canonical shared period model for this collection."""
        self._period_ids = period_ids
        self._weights = weights
        self._num_periods = num_periods
        for stream in self._streams.values():
            stream.set_period_context(
                weights=self._weights,
                period_ids=self._period_ids,
                num_periods=self._num_periods,
            )
        self._invalidate_numeric_cache()

    def _validate_stream_period_context(self, stream: "Stream") -> None:
        if (
            stream.num_periods == self._num_periods
            or stream.num_periods == 1
            or self._num_periods == 1
        ):
            return
        raise ValueError(
            f"weights for stream '{stream.name}' must align with "
            "the collection to be added."
        )

    def _adopt_appropriate_period_context(
        self, other: "Stream", obj: "StreamCollection" | "Stream"
    ) -> None:
        self_num_periods = self._num_periods or 0
        other_num_periods = other._num_periods or 0
        if self_num_periods >= other_num_periods:
            obj.set_period_context(
                period_ids=self._period_ids,
                weights=self._weights,
                num_periods=self._num_periods,
            )
        else:
            if obj is not other and obj is not None:
                obj.set_period_context(
                    period_ids=other._period_ids,
                    weights=other._weights,
                    num_periods=other._num_periods,
                )
            self.set_period_context(
                period_ids=other._period_ids,
                weights=other._weights,
                num_periods=other._num_periods,
            )

    def numeric_view(self, idx: int | None = None) -> StreamCollectionNumericView:
        """Return a cached dense numeric view for stream-analysis kernels."""
        period_idx = None if idx is None else int(idx)
        signature = self._numeric_signature()
        cache_key = ("parent", period_idx, signature)
        cached = self._numeric_cache.get(cache_key)
        if cached is not None:
            return cached

        view = self._build_numeric_view(period_idx)
        self._numeric_cache.clear()
        self._numeric_cache[cache_key] = view
        return view

    def segment_numeric_view(
        self,
        idx: int | None = None,
    ) -> StreamCollectionNumericView:
        """Return a cached numeric view expanded to ordered thermal segments."""
        period_idx = None if idx is None else int(idx)
        signature = self._numeric_signature()
        cache_key = ("segment", period_idx, signature)
        cached = self._numeric_cache.get(cache_key)
        if cached is not None:
            return cached
        view = build_segment_numeric_view(
            list(self._streams.values()),
            period_idx,
            keys=list(self._streams),
        )
        self._numeric_cache.clear()
        self._numeric_cache[cache_key] = view
        return view

    def _numeric_signature(self) -> tuple:
        return tuple(
            (
                id(stream),
                int(getattr(stream, "_numeric_revision", 0)),
                tuple(
                    (id(segment), int(getattr(segment, "_numeric_revision", 0)))
                    for segment in getattr(stream, "segments", ())
                ),
            )
            for stream in self._streams.values()
        )

    def _build_numeric_view(
        self, idx: int | None = None
    ) -> StreamCollectionNumericView:
        return build_numeric_view(
            list(self._streams.values()),
            idx,
            keys=list(self._streams),
        )

    @staticmethod
    def _value_at_idx(value, idx: int | None = None) -> float:
        return value_at_idx(value, idx)

    def _invalidate_numeric_cache(self) -> None:
        self._numeric_cache.clear()

    def get_index(self, stream) -> int:
        """Return the position (index) of a stream object in the sorted stream list."""
        self._ensure_sorted()
        for idx, s in enumerate(self._sorted_cache):
            if s == stream:
                return idx
        raise ValueError("Stream not found in collection.")

    def _ensure_sorted(self):
        """(Internal) Sort streams if needed."""
        if self._needs_sort:
            self._sorted_cache = sorted(
                self._streams.values(),
                key=self._sort_key,
                reverse=self._sort_reverse,
            )
            self._needs_sort = False

    def items(self):
        """Return the underlying keyed stream items in insertion order."""
        return self._streams.items()

    def __iter__(self):
        self._ensure_sorted()
        return iter(self._sorted_cache)

    def __add__(self, other: StreamCollection) -> StreamCollection:
        if not isinstance(other, StreamCollection):
            return NotImplemented
        combined = StreamCollection()
        if self._period_ids is not None:
            combined.set_period_context(
                self._period_ids, self._weights, self._num_periods
            )
        elif other._period_ids is not None:
            combined.set_period_context(
                other._period_ids, other._weights, other._num_periods
            )
        if (
            self._period_ids is not None
            and other._period_ids is not None
            and self._period_ids != other._period_ids
            and self._num_periods > 1
            and other._num_periods > 1
        ):
            raise ValueError(
                "Cannot combine StreamCollections with different period_ids."
            )
        else:
            self._adopt_appropriate_period_context(other, combined)

        # Add all streams from self
        for key, stream in self._streams.items():
            combined.add(stream=stream, key=key)
        # Add all streams from other
        for key, stream in other._streams.items():
            combined.add(stream=stream, key=key)
        return combined

    def __len__(self):
        return len(self._streams)

    def __getitem__(self, key):
        if isinstance(key, int):
            self._ensure_sorted()
            try:
                return self._sorted_cache[key]
            except IndexError as exc:
                raise IndexError(
                    f"Stream index {key} out of range for collection of size "
                    f"{len(self._sorted_cache)}."
                ) from exc
        if isinstance(key, str):
            return self._streams[key]
        raise TypeError(
            f"Invalid key type {type(key)}. Must be str (name) or int (index)."
        )

    def __contains__(self, stream_name: str):
        return stream_name in self._streams

    def __repr__(self):
        return f"StreamCollection({list(self._streams.keys())})"

    def __eq__(self, other):
        if not isinstance(other, StreamCollection):
            return NotImplemented
        return self._streams == other._streams

    def __getstate__(self):
        state = self.__dict__.copy()
        mode, sort_detail = state.get("_sort_spec", ("attr", "t_supply"))
        if mode == "callable" and not _is_picklable(sort_detail):
            state["_sort_spec"] = ("attr", "t_supply")
        state["_sort_key"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "_numeric_cache" not in self.__dict__:
            self._numeric_cache = {}
        self._rebuild_sort_key()

    def to_dict(
        self,
        idx: int | None = None,
        *,
        expand_segments: bool = False,
    ) -> dict[str, list[Any]]:
        """Return stream data as serializable rows in standard reporting order."""
        return collection_to_dict(
            self,
            idx=idx,
            expand_segments=expand_segments,
        )

    @staticmethod
    def _descending_sort_value(value: float) -> float:
        if math.isnan(value):
            return math.inf
        return -value

    @classmethod
    def _dict_category(cls, stream: "Stream") -> str:
        if stream.is_process_stream and stream.type == ST.Hot.value:
            return "hot_stream"
        if stream.is_process_stream and stream.type == ST.Cold.value:
            return "cold_stream"
        if not stream.is_process_stream and stream.type == ST.Hot.value:
            return "hot_utility"
        if not stream.is_process_stream and stream.type == ST.Cold.value:
            return "cold_utility"
        return "other"

    @classmethod
    def _dict_sort_key(cls, stream: "Stream", idx: int | None = None) -> tuple:
        category_order = {
            "hot_stream": 0,
            "cold_stream": 1,
            "hot_utility": 2,
            "cold_utility": 3,
            "other": 4,
        }
        category = cls._dict_category(stream)
        supply = cls._descending_sort_value(value_at_idx(stream._t_supply, idx))
        target = cls._descending_sort_value(value_at_idx(stream._t_target, idx))
        if category == "hot_stream":
            return (category_order[category], supply, target, stream.name)
        return (category_order[category], supply, stream.name)

    # === Filtered StreamCollection subset builders ===
    def _build_stream_subset(
        self,
        target_type: str | None,
        include_process_streams: bool = True,
        include_utility_streams: bool = True,
        invert_utility: bool = False,
        sort_attr: str | None = None,
    ) -> "StreamCollection":
        return build_stream_subset(
            self,
            target_type=target_type,
            include_process_streams=include_process_streams,
            include_utility_streams=include_utility_streams,
            invert_utility=invert_utility,
            sort_attr=sort_attr,
        )

    def get_hot_streams(
        self,
        include_process_streams: bool = True,
        include_utility_streams: bool = True,
        invert_utility: bool = False,
        sort_attr: str | None = None,
    ):
        """Return a new collection containing only hot streams."""
        return self._build_stream_subset(
            target_type=ST.Hot.value,
            include_process_streams=include_process_streams,
            include_utility_streams=include_utility_streams,
            invert_utility=invert_utility,
            sort_attr=sort_attr,
        )

    def get_cold_streams(
        self,
        include_process_streams: bool = True,
        include_utility_streams: bool = True,
        invert_utility: bool = False,
        sort_attr: str | None = None,
    ):
        """Return a new collection containing only cold streams."""
        return self._build_stream_subset(
            target_type=ST.Cold.value,
            include_process_streams=include_process_streams,
            include_utility_streams=include_utility_streams,
            invert_utility=invert_utility,
            sort_attr=sort_attr,
        )

    def get_process_streams(self, sort_attr: str | None = None):
        """Return a new collection containing only process streams."""
        return self._build_stream_subset(
            target_type=None,
            include_process_streams=True,
            include_utility_streams=False,
            invert_utility=False,
            sort_attr=sort_attr,
        )

    def get_hot_process_streams(self, sort_attr: str | None = None):
        """Return a new collection containing only hot process streams."""
        return self._build_stream_subset(
            target_type=ST.Hot.value,
            include_process_streams=True,
            include_utility_streams=False,
            invert_utility=False,
            sort_attr=sort_attr,
        )

    def get_cold_process_streams(self, sort_attr: str | None = None):
        """Return a new collection containing only cold process streams."""
        return self._build_stream_subset(
            target_type=ST.Cold.value,
            include_process_streams=True,
            include_utility_streams=False,
            invert_utility=False,
            sort_attr=sort_attr,
        )

    def get_utility_streams(self, sort_attr: str | None = None):
        """Return a new collection containing only utility streams."""
        return self._build_stream_subset(
            target_type=None,
            include_process_streams=False,
            include_utility_streams=True,
            invert_utility=False,
            sort_attr=sort_attr,
        )

    def get_hot_utility_streams(self, sort_attr: str | None = None):
        """Return a new collection containing only hot utility streams."""
        return self._build_stream_subset(
            target_type=ST.Hot.value,
            include_process_streams=False,
            include_utility_streams=True,
            invert_utility=False,
            sort_attr=sort_attr,
        )

    def get_cold_utility_streams(self, sort_attr: str | None = None):
        """Return a new collection containing only cold utility streams."""
        return self._build_stream_subset(
            target_type=ST.Cold.value,
            include_process_streams=False,
            include_utility_streams=True,
            invert_utility=False,
            sort_attr=sort_attr,
        )

    def get_inverted_hot_utility_streams(self, sort_attr: str | None = None):
        """Return a new collection containing only inverted hot utility streams."""
        return self._build_stream_subset(
            target_type=ST.Hot.value,
            include_process_streams=False,
            include_utility_streams=True,
            invert_utility=True,
            sort_attr=sort_attr,
        )

    def get_inverted_cold_utility_streams(self, sort_attr: str | None = None):
        """Return a new collection containing only inverted cold utility streams."""
        return self._build_stream_subset(
            target_type=ST.Cold.value,
            include_process_streams=False,
            include_utility_streams=True,
            invert_utility=True,
            sort_attr=sort_attr,
        )
