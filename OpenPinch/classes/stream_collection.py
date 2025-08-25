from typing import Callable, Dict, List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..classes import Stream, Stream


class StreamCollection:
    """A dynamic, ordered collection of streams.

    Features:
        - Add and remove streams by name.
        - Prevent overwriting existing streams by auto-renaming.
        - Set custom sort keys (attribute name, list of attributes, or callable).
        - Supports efficient iteration with lazy sorting.
        - Allows ascending or descending sorting.

    Typical usage:
        - Store and manage process streams or utility streams.
        - Sort streams dynamically by attributes like temperature or flow.
        - Avoid duplicate names automatically.

    Example:
        zone = StreamCollection()
        zone.add(Stream("H1", 300, 400))
        zone.set_sort_key(["t_target", "t_supply"], reverse=True)
        for stream in zone:
            print(stream.name, stream.t_target)
    """

    def __init__(self):
        self._streams: Dict[str, object] = {}
        self._sort_key: Callable = lambda s: s.t_supply  # default: sort by name
        self._sort_reverse: bool = True
        self._sorted_cache: List[object] = []
        self._needs_sort: bool = True

    def add(self, stream, key: str = None, prevent_overwrite: bool = True):
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

    def add_many(self, streams, keys = None, prevent_overwrite: bool = True):
        if keys == None:
            for stream in streams:
                self.add(stream, prevent_overwrite=prevent_overwrite)
        else:
            if len(streams) != len(keys):
                raise ValueError("Length of streams and keys must match.")
            for stream, key in zip(streams, keys):
                self.add(stream, key, prevent_overwrite)

    def replace(self, stream_dict: Dict[str, Union["Stream", "Stream"]]):
        self._streams = {}
        for stream in stream_dict.values():
            self._streams[stream.name] = stream
        self._needs_sort = True

    def remove(self, stream_name: str):
        if stream_name in self._streams:
            del self._streams[stream_name]
            self._needs_sort = True
        else:
            raise KeyError(f"Stream '{stream_name}' not found.")

    def set_sort_key(self, key: Union[str, List[str], Callable], reverse: bool = False):
        """Set the sorting key. Supports attribute names or custom lambdas."""
        self._sort_reverse = reverse
        if isinstance(key, str):
            self._sort_key = lambda s: getattr(s, key)
        elif isinstance(key, list):
            self._sort_key = lambda s: tuple(
                -getattr(s, attr) if reverse else getattr(s, attr) for attr in key
            )
        else:
            self._sort_key = key
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
            # Allow indexing by integer
            return list(self._streams.values())[key]
        elif isinstance(key, str):
            # Allow accessing by stream name
            return self._streams[key]
        else:
            raise TypeError(f"Invalid key type {type(key)}. Must be str (name) or int (index).")

    def __contains__(self, stream_name: str):
        return stream_name in self._streams

    def __repr__(self):
        return f"StreamCollection({list(self._streams.keys())})"
    
    def __eq__(self, other):
        if not isinstance(other, StreamCollection):
            return NotImplemented
        return self._streams == other._streams
