"""Utility container for managing ordered sets of stream objects."""

import csv
import pickle
import warnings
from copy import copy, deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Callable, List, Tuple, Union

import numpy as np

from ..lib.enums import ST
from .stream import Stream
from .value import Value


def _sort_by_attr(attr: str, stream: object):
    return getattr(stream, attr)


def _sort_by_attrs(attrs: Tuple[str, ...], stream: object):
    return tuple(getattr(stream, attr) for attr in attrs)


def _stream_attr_value(stream: object, attr_name: str, idx: int | None = None):
    value = getattr(stream, attr_name)
    if isinstance(value, Value):
        state_idx = 0 if idx is None else int(idx)
        return float(value[state_idx])
    return value


def _is_picklable(obj: object) -> bool:
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


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
        self._state_ids: dict[str, int] | None = {"0": 0}
        self._weights: np.ndarray | None = np.array([1.0])
        self._sort_spec: Tuple[str, Any] = ("attr", "t_supply")
        self._sort_key: Callable = partial(_sort_by_attr, "t_supply")
        self._sort_reverse: bool = True
        self._sorted_cache: List[object] = []
        self._needs_sort: bool = True
        self._num_states: int | None = 1
        if streams is not None:
            self.add_many(streams)

    @property
    def state_ids(self) -> dict[str, int] | None:
        """Return the canonical state identifiers for this collection."""
        return self._state_ids

    @property
    def weights(self) -> np.ndarray | None:
        """Return the canonical state weights for this collection."""
        return self._weights

    @property
    def num_states(self) -> int | None:
        """Return the number of states for this collection."""
        return self._num_states

    def _rebuild_sort_key(self):
        mode, payload = self._sort_spec
        if mode == "attr":
            self._sort_key = partial(_sort_by_attr, payload)
        elif mode == "attrs":
            self._sort_key = partial(_sort_by_attrs, payload)
        else:
            self._sort_key = payload

    def add(
        self, stream: "Stream", key: str = None, prevent_overwrite: bool = True
    ) -> str:
        """Insert a stream, optionally renaming the key to avoid collisions."""
        self._validate_stream_state_context(stream)
        self._adopt_appropriate_state_context(stream, stream)
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

        stream.set_state_context(
            weights=self._weights,
            state_ids=self._state_ids,
            num_states=self._num_states,
        )
        self._needs_sort = True
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
                stream.set_value_attr_at_state_idx(attr_name, value, idx=idx)
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

    def set_state_context(
        self,
        state_ids: dict[str, int] | list[str] | tuple[str, ...] | None,
        weights: np.ndarray | list[float] | tuple[float, ...] | None,
        num_states: int | None = None,
    ) -> None:
        """Persist the canonical shared state model for this collection."""
        self._state_ids = state_ids
        self._weights = weights
        self._num_states = num_states
        for stream in self._streams.values():
            stream.set_state_context(
                weights=self._weights,
                state_ids=self._state_ids,
                num_states=self._num_states,
            )

    def _validate_stream_state_context(self, stream: "Stream") -> None:
        if (
            stream.num_states == self._num_states
            or stream.num_states == 1
            or self._num_states == 1
        ):
            return
        raise ValueError(
            f"weights for stream '{stream.name}' must align with "
            "the collection to be added."
        )

    def _adopt_appropriate_state_context(
        self, other: "Stream", obj: "StreamCollection" | "Stream"
    ) -> None:
        if self._num_states >= other._num_states:
            obj.set_state_context(
                state_ids=self._state_ids,
                weights=self._weights,
                num_states=self._num_states,
            )
        else:
            if obj is not other and obj is not None:
                obj.set_state_context(
                    state_ids=other._state_ids,
                    weights=other._weights,
                    num_states=other._num_states,
                )
            self.set_state_context(
                state_ids=other._state_ids,
                weights=other._weights,
                num_states=other._num_states,
            )

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
        if self._state_ids is not None:
            combined.set_state_context(self._state_ids, self._weights, self._num_states)
        elif other._state_ids is not None:
            combined.set_state_context(
                other._state_ids, other._weights, other._num_states
            )
        if (
            self._state_ids is not None
            and other._state_ids is not None
            and self._state_ids != other._state_ids
            and self._num_states > 1
            and other._num_states > 1
        ):
            raise ValueError(
                "Cannot combine StreamCollections with different state_ids."
            )
        else:
            self._adopt_appropriate_state_context(other, combined)

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
        mode, payload = state.get("_sort_spec", ("attr", "t_supply"))
        if mode == "callable" and not _is_picklable(payload):
            state["_sort_spec"] = ("attr", "t_supply")
        state["_sort_key"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._rebuild_sort_key()

    def export_to_csv(self, filename: str = "heat pump streams") -> Path:
        """Export stream data to ``results/<filename>`` and return the path written."""
        base_dir = Path(__file__).resolve().parents[2] / "results"
        base_dir.mkdir(parents=True, exist_ok=True)
        output_path = base_dir / (filename + ".csv")

        self._ensure_sorted()
        with output_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "name",
                    "t_supply",
                    "t_target",
                    "heat_flow",
                    "dt_cont",
                    "dt_cont_multiplier",
                    "htc",
                ]
            )
            for stream in self._sorted_cache:
                writer.writerow(
                    [
                        stream.name,
                        stream.t_supply,
                        stream.t_target,
                        stream.heat_flow,
                        stream.dt_cont,
                        stream.dt_cont_multiplier,
                        stream.htc,
                    ]
                )

        return output_path

    # === Filtered StreamCollection subset builders ===
    def _build_stream_subset(
        self,
        target_type: str | None,
        include_process_streams: bool = True,
        include_utility_streams: bool = True,
        invert_utility: bool = False,
        sort_attr: str | None = None,
    ) -> "StreamCollection":
        if invert_utility:
            include_process_streams = False
            include_utility_streams = True

        subset = StreamCollection()
        subset._state_ids = self._state_ids
        subset._weights = self._weights
        subset._sort_spec = self._sort_spec
        subset._rebuild_sort_key()
        subset._sort_reverse = self._sort_reverse

        for key, stream in self._streams.items():
            if stream.is_process_stream:
                if not include_process_streams:
                    continue
                if target_type is None or stream.type == target_type:
                    subset._streams[key] = stream
                continue

            if not include_utility_streams:
                continue

            if invert_utility:
                opposite_type = (
                    ST.Cold.value if target_type == ST.Hot.value else ST.Hot.value
                )
                if stream.type != opposite_type:
                    continue
                inverted_stream = copy(stream)
                inverted_stream.invert()
                subset._streams[key] = inverted_stream
            elif target_type is None or stream.type == target_type:
                subset._streams[key] = stream

        if sort_attr is None:
            subset._sort_spec = self._sort_spec
            subset._rebuild_sort_key()
            subset._sort_reverse = self._sort_reverse
        else:
            subset.set_sort_key(sort_attr, reverse=self._sort_reverse)
        subset._needs_sort = True
        return subset

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
