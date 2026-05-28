"""Utility container for managing ordered sets of stream objects."""

import csv
import pickle
import warnings
from copy import copy, deepcopy
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Union

import numpy as np

from ..lib.enums import ST

if TYPE_CHECKING:
    from .stream import Stream


def _sort_by_attr(attr: str, stream: object):
    return getattr(stream, attr)


def _sort_by_attrs(attrs: Tuple[str, ...], stream: object):
    return tuple(getattr(stream, attr) for attr in attrs)


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

    def __init__(self):
        """Initialise an empty collection sorted by descending supply temperature."""
        self._streams: Dict[str, object] = {}
        self._sort_spec: Tuple[str, Any] = ("attr", "t_supply")
        self._sort_key: Callable = partial(_sort_by_attr, "t_supply")
        self._sort_reverse: bool = True
        self._sorted_cache: List[object] = []
        self._needs_sort: bool = True

    def _rebuild_sort_key(self):
        mode, payload = self._sort_spec
        if mode == "attr":
            self._sort_key = partial(_sort_by_attr, payload)
        elif mode == "attrs":
            self._sort_key = partial(_sort_by_attrs, payload)
        else:
            self._sort_key = payload

    def add(self, stream, key: str = None, prevent_overwrite: bool = True) -> str:
        """Insert a stream, optionally renaming the key to avoid collisions."""
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
        self._needs_sort = True
        return key

    def add_many(self, streams, keys=None, prevent_overwrite: bool = True):
        """Insert several streams, optionally using explicit keys for each stream."""
        if keys is None:
            for stream in streams:
                self.add(stream, prevent_overwrite=prevent_overwrite)
        else:
            if len(streams) != len(keys):
                raise ValueError("Length of streams and keys must match.")
            for stream, key in zip(streams, keys):
                self.add(stream, key, prevent_overwrite)

    def _build_stream_subset(
        self,
        target_type: str | None,
        include_process_streams: bool = True,
        include_utility_streams: bool = True,
        invert_utility: bool = False,
    ) -> "StreamCollection":
        if invert_utility:
            include_process_streams = False
            include_utility_streams = True

        subset = StreamCollection()
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

        subset._needs_sort = True
        return subset

    def get_hot_streams(
        self,
        include_process_streams: bool = True,
        include_utility_streams: bool = True,
        invert_utility: bool = False,
    ):
        """Return a new collection containing only hot streams."""
        return self._build_stream_subset(
            target_type=ST.Hot.value,
            include_process_streams=include_process_streams,
            include_utility_streams=include_utility_streams,
            invert_utility=invert_utility,
        )

    def get_cold_streams(
        self,
        include_process_streams: bool = True,
        include_utility_streams: bool = True,
        invert_utility: bool = False,
    ):
        """Return a new collection containing only cold streams."""
        return self._build_stream_subset(
            target_type=ST.Cold.value,
            include_process_streams=include_process_streams,
            include_utility_streams=include_utility_streams,
            invert_utility=invert_utility,
        )

    def get_process_streams(self):
        """Return a new collection containing only process streams."""
        return self._build_stream_subset(
            target_type=None,
            include_process_streams=True,
            include_utility_streams=False,
            invert_utility=False,
        )

    def get_hot_process_streams(self):
        """Return a new collection containing only hot process streams."""
        return self._build_stream_subset(
            target_type=ST.Hot.value,
            include_process_streams=True,
            include_utility_streams=False,
            invert_utility=False,
        )

    def get_cold_process_streams(self):
        """Return a new collection containing only cold process streams."""
        return self._build_stream_subset(
            target_type=ST.Cold.value,
            include_process_streams=True,
            include_utility_streams=False,
            invert_utility=False,
        )

    def get_utility_streams(self):
        """Return a new collection containing all utility streams."""
        return self._build_stream_subset(
            target_type=None,
            include_process_streams=False,
            include_utility_streams=True,
            invert_utility=False,
        )

    def get_hot_utility_streams(self):
        """Return a new collection containing only hot utility streams."""
        subset = self._build_stream_subset(
            target_type=ST.Hot.value,
            include_process_streams=False,
            include_utility_streams=True,
            invert_utility=False,
        )
        subset.set_sort_key(lambda stream: stream.t_supply.max, reverse=True)
        return subset

    def get_cold_utility_streams(self):
        """Return a new collection containing only cold utility streams."""
        subset = self._build_stream_subset(
            target_type=ST.Cold.value,
            include_process_streams=False,
            include_utility_streams=True,
            invert_utility=False,
        )
        subset.set_sort_key(lambda stream: stream.t_supply.max, reverse=True)
        return subset

    def get_inverted_hot_utility_streams(self):
        """Return a new collection containing only hot utility streams."""
        return self._build_stream_subset(
            target_type=ST.Hot.value,
            include_process_streams=False,
            include_utility_streams=True,
            invert_utility=True,
        )

    def get_inverted_cold_utility_streams(self):
        """Return a new collection containing only cold streams."""
        return self._build_stream_subset(
            target_type=ST.Cold.value,
            include_process_streams=False,
            include_utility_streams=True,
            invert_utility=True,
        )

    def get_stream_by_name(self, name: str, approximate: bool = False) -> Stream:
        for stream in self:
            if (stream.name == name) or (approximate and name in stream.name):
                return stream
        warnings.warn(f"Stream '{name}' not found.")
        return None

    def replace(self, stream_dict: Dict[str, Union["Stream", "Stream"]]):
        """Replace the collection contents with the provided stream mapping."""
        self._streams = {}
        for stream in stream_dict.values():
            self._streams[stream.name] = stream
        self._needs_sort = True

    def remove(self, stream_name: str):
        """Remove a stream by name."""
        if stream_name in self._streams:
            del self._streams[stream_name]
            self._needs_sort = True
        else:
            warnings.warn(f"Stream '{stream_name}' not found.")

    def sum_stream_attribute(self, attr_name: str) -> float:
        """Return the total of a specified attribute for streams in the collection."""
        if self._streams is None or len(self._streams) == 0:
            warnings.warn(
                f"Attempted to sum attribute '{attr_name}'"
                f" on an empty stream collection."
            )
            return 0.0
        stream = next(iter(self._streams.values()))
        if hasattr(stream, attr_name):
            return sum(getattr(stream, attr_name) for stream in self._streams.values())
        else:
            warnings.warn(
                f"Stream '{stream.name}' does not have attribute '{attr_name}'."
            )
        return 0.0

    def set_common_stream_attribute(self, attr_name: str, value: Any):
        """Set a common attribute across all streams in the collection."""
        if self._streams is None or len(self._streams) == 0:
            warnings.warn(
                f"Attempted to set attribute '{attr_name}'"
                f"on an empty stream collection."
            )
            return 0.0
        for stream in self._streams.values():  # Check if attribute exists
            if hasattr(stream, attr_name):
                current_value = getattr(stream, attr_name)
                if current_value != value:
                    setattr(stream, attr_name, value)
            else:
                warnings.warn(
                    f"Stream '{stream.name}' does not have attribute '{attr_name}'."
                )
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
        sort_key: Union[str, List[str], Callable, None] = None,
        reverse: bool | None = None,
    ) -> "StreamCollection":
        """Return a copy of the collection, optionally deep-copying streams.

        By default the new collection preserves the current sort configuration.
        Callers may override that with ``sort_key`` and optionally ``reverse``.
        """
        copied = StreamCollection()
        if sort_key is None:
            copied._sort_spec = self._sort_spec
            copied._rebuild_sort_key()
            copied._sort_reverse = self._sort_reverse if reverse is None else reverse
        else:
            copied.set_sort_key(
                sort_key,
                reverse=self._sort_reverse if reverse is None else reverse,
            )

        for key, stream in self._streams.items():
            copied.add(
                deepcopy(stream) if deep else stream,
                key=key,
                prevent_overwrite=False,
            )

        return copied

    def validate_state_alignment(
        self,
        *,
        allow_scalar_broadcast: bool = True,
    ) -> tuple[list[str] | None, np.ndarray | None]:
        """Validate that all explicitly stateful streams share one state model."""
        canonical_key: str | None = None
        canonical_state_ids: list[str] | None = None
        canonical_weights: np.ndarray | None = None

        for key, stream in self._streams.items():
            state_ids, weights = stream._get_state_context()
            if state_ids is None:
                if allow_scalar_broadcast:
                    continue
                raise ValueError(
                    f"Stream '{key}' is scalar but scalar broadcast is disabled."
                )

            if canonical_state_ids is None:
                canonical_key = key
                canonical_state_ids = state_ids
                canonical_weights = weights
                continue

            if state_ids != canonical_state_ids:
                raise ValueError(
                    f"state_ids for stream '{key}' must align with '{canonical_key}'."
                )
            if not np.allclose(
                weights,
                canonical_weights,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError(
                    f"weights for stream '{key}' must align with '{canonical_key}'."
                )

        return (
            canonical_state_ids,
            None if canonical_weights is None else canonical_weights.copy(),
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
                self._streams.values(), key=self._sort_key, reverse=self._sort_reverse
            )
            self._needs_sort = False

    def __iter__(self):
        self._ensure_sorted()
        return iter(self._sorted_cache)

    def items(self):
        """Return the underlying keyed stream items in insertion order."""
        return self._streams.items()

    def __add__(self, other):
        if not isinstance(other, StreamCollection):
            return NotImplemented
        combined = StreamCollection()
        # Add all streams from self
        for stream in self._streams.values():
            combined.add(stream)
        # Add all streams from other
        for stream in other._streams.values():
            combined.add(stream)
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
        elif isinstance(key, str):
            # Allow accessing by stream name
            return self._streams[key]
        else:
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
                    "dt_cont_act",
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
                        stream.dt_cont_act,
                        stream.htc,
                    ]
                )

        return output_path
