"""Utility container for managing ordered sets of stream objects."""

import csv
import pickle
from pathlib import Path
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Union

from ..lib.enums import *

if TYPE_CHECKING:
    from ..classes import Stream


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

    def add(self, stream, key: str = None, prevent_overwrite: bool = True):
        """Insert a stream, optionally renaming the key to avoid collisions."""
        if key is None:
            key = stream.name
        original_key = key
        counter = 1
        while prevent_overwrite and key in self._streams:
            key = f"{original_key}_{counter}"
            counter += 1
        # stream.name = key
        self._streams[key] = stream
        self._needs_sort = True

    def add_many(self, streams, keys=None, prevent_overwrite: bool = True):
        """Insert several streams, optionally using explicit keys for each stream."""
        if keys == None:
            for stream in streams:
                self.add(stream, prevent_overwrite=prevent_overwrite)
        else:
            if len(streams) != len(keys):
                raise ValueError("Length of streams and keys must match.")
            for stream, key in zip(streams, keys):
                self.add(stream, key, prevent_overwrite)

    def get_hot_streams(self):
        """Return a new collection containing only hot streams."""
        hot_streams = StreamCollection()
        hot_streams._sort_spec = self._sort_spec
        hot_streams._rebuild_sort_key()
        hot_streams._sort_reverse = self._sort_reverse
        hot_streams._streams = {
            key: stream
            for key, stream in self._streams.items()
            if stream.type == StreamType.Hot.value
        }
        hot_streams._needs_sort = True
        return hot_streams

    def get_cold_streams(self):
        """Return a new collection containing only cold streams."""
        cold_streams = StreamCollection()
        cold_streams._sort_spec = self._sort_spec
        cold_streams._rebuild_sort_key()
        cold_streams._sort_reverse = self._sort_reverse
        cold_streams._streams = {
            key: stream
            for key, stream in self._streams.items()
            if stream.type == StreamType.Cold.value
        }
        cold_streams._needs_sort = True
        return cold_streams

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
            raise KeyError(f"Stream '{stream_name}' not found.")

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
                    f"Stream index {key} out of range for collection of size {len(self._sorted_cache)}."
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
                ["name", "t_supply", "t_target", "heat_flow", "dt_cont", "htc"]
            )
            for stream in self._sorted_cache:
                writer.writerow(
                    [
                        stream.name,
                        stream.t_supply,
                        stream.t_target,
                        stream.heat_flow,
                        stream.dt_cont,
                        stream.htc,
                    ]
                )

        return output_path
