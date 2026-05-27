"""Zone data structure capturing nested scopes and their thermal targets."""

import warnings
from typing import TYPE_CHECKING, Optional

from ..lib.config import Configuration
from ..lib.enums import ZT
from ..lib.schemas.targets import BaseTargetModel
from .stream_collection import StreamCollection

if TYPE_CHECKING:
    from .stream import Stream
    from .zone import Zone


class Zone:
    """Hierarchical analysis boundary containing streams, utilities, and targets.

    Zones form the backbone of the in-memory OpenPinch model. Each zone can own
    process streams, utility streams, solved targets, generated graphs, and
    nested child zones. Direct and indirect integration routines progressively
    populate this structure as the analysis moves from local process scopes up
    to site-style aggregation.
    """

    def __init__(
        self,
        name: str = "Zone",
        type: str = ZT.P.value,
        zone_config: Optional[Configuration] = None,
        parent_zone: "Zone" = None,
    ):
        """Initialise an empty zone with stream, target, and graph containers."""
        # === Metadata ===
        self._name = name
        self._type = type
        self._config = zone_config or Configuration()
        self._parent_zone = parent_zone
        self._dt_cont_multiplier = (
            parent_zone.dt_cont_multiplier
            if hasattr(parent_zone, "dt_cont_multiplier")
            else 1.0
        )
        self._lock_dt_cont_multiplier = False
        self._active = True
        self._subzones = {}
        self._targets = {}
        self._graphs = {}

        # === Streams & Utilities ===
        self._hot_streams: StreamCollection = StreamCollection()
        self._cold_streams: StreamCollection = StreamCollection()
        self._net_hot_streams: StreamCollection = StreamCollection()
        self._net_cold_streams: StreamCollection = StreamCollection()
        self._hot_utilities: StreamCollection = StreamCollection()
        self._cold_utilities: StreamCollection = StreamCollection()

    # === Properties ===

    @property
    def name(self):
        """Display name used when addressing the zone in the hierarchy."""
        return self._name

    @name.setter
    def name(self, value):
        """Set the display name used for this zone and its address."""
        self._name = value

    @property
    def type(self):
        """Zone type type from :class:`ZoneType`."""
        return self._type

    @type.setter
    def type(self, value):
        """Set the zone classification used by hierarchical targeting logic."""
        self._type = value

    @property
    def config(self):
        """Configuration object controlling analysis behaviour for this zone."""
        return self._config

    @config.setter
    def config(self, value):
        """Replace the runtime configuration attached to this zone."""
        self._config = value

    @property
    def parent_zone(self):
        """Direct parent zone in the site hierarchy, if any."""
        return self._parent_zone

    @parent_zone.setter
    def parent_zone(self, value):
        """Attach the zone to a different parent in the hierarchy."""
        self._parent_zone = value

    @property
    def active(self) -> bool:
        """Whether the zone participates in the current analysis."""
        return bool(self._active)

    @active.setter
    def active(self, value: bool):
        """Activate or deactivate the zone for subsequent analysis passes."""
        self._active = bool(value)

    @property
    def address(self) -> str:
        """Slash-delimited path from the root zone to this zone."""
        if self.parent_zone is None:
            return str(self.name)
        return f"{self.parent_zone.address}/{self.name}"

    @property
    def dt_cont_multiplier(self) -> float:
        """Effective multiplier applied to stream and utility ``dt_cont`` values."""
        return self._dt_cont_multiplier

    @dt_cont_multiplier.setter
    def dt_cont_multiplier(self, value: float):
        """Set the active ``dt_cont`` multiplier when the zone is still mutable."""
        for zone in self.subzones.values():
            if not (zone._lock_dt_cont_multiplier):
                zone.dt_cont_multiplier = value
        self._dt_cont_multiplier = float(value)
        self.all_streams.set_common_stream_attribute("dt_cont_multiplier", value)
        self._targets.clear()

    @property
    def hot_streams(self):
        """Process streams that release heat within this zone."""
        return self._hot_streams

    @hot_streams.setter
    def hot_streams(self, data):
        """Replace the zone hot-stream collection."""
        self._hot_streams = data

    @property
    def cold_streams(self):
        """Process streams that require heat within this zone."""
        return self._cold_streams

    @cold_streams.setter
    def cold_streams(self, data):
        """Replace the zone cold-stream collection."""
        self._cold_streams = data

    @property
    def net_hot_streams(self):
        """Net hot streams derived from zonal aggregation."""
        return self._net_hot_streams

    @net_hot_streams.setter
    def net_hot_streams(self, data):
        """Replace the aggregated net hot-stream collection."""
        self._net_hot_streams = data

    @property
    def net_cold_streams(self):
        """Net cold streams derived from zonal aggregation."""
        return self._net_cold_streams

    @net_cold_streams.setter
    def net_cold_streams(self, data):
        """Replace the aggregated net cold-stream collection."""
        self._net_cold_streams = data

    @property
    def hot_utilities(self):
        """Hot utility streams assigned to the zone."""
        return self._hot_utilities

    @hot_utilities.setter
    def hot_utilities(self, data):
        """Replace the zone hot-utility collection."""
        self._hot_utilities = data

    @property
    def cold_utilities(self):
        """Cold utility streams assigned to the zone."""
        return self._cold_utilities

    @cold_utilities.setter
    def cold_utilities(self, data):
        """Replace the zone cold-utility collection."""
        self._cold_utilities = data

    @property
    def graphs(self):
        """Graphs generated for this zone."""
        return self._graphs

    @graphs.setter
    def graphs(self, data):
        """Replace the graph payloads cached on this zone."""
        self._graphs = data

    @property
    def subzones(self):
        """Immediate child zones keyed by name."""
        return self._subzones

    @property
    def targets(self):
        """Energy targets keyed by target name."""
        return self._targets

    @property
    def process_streams(self):
        """Combined hot and cold process streams for the zone."""
        return self._hot_streams + self._cold_streams

    @property
    def net_process_streams(self):
        """Combined net hot and net cold process streams for the zone."""
        return self._net_hot_streams + self._net_cold_streams

    @property
    def utility_streams(self):
        """Combined hot and cold utility streams for the zone."""
        return self._hot_utilities + self._cold_utilities

    @property
    def all_streams(self):
        """All process and utility streams defined on the zone."""
        return self.process_streams + self.utility_streams

    @property
    def all_net_streams(self):
        """All net-process and utility streams defined on the zone."""
        return self.net_process_streams + self.utility_streams

    # === Methods ===
    def add_graph(self, name: str, result):
        """Store a graph result under ``name`` for later export or display."""
        self._graphs[name] = result

    def add_zone(self, zone_to_add, sub: bool = True):
        """Add a single zone object keyed by its name.

        If the zone name already exists:
        - If the zone is identical (e.g. same stream and utility objects), skip.
        - If it's different, add it with a suffix like '_1', '_2', etc.
        """
        base_name = getattr(zone_to_add, "name", None)

        if not isinstance(base_name, str):
            raise ValueError(
                "Zone must have a string 'name' attribute, got: "
                f"{type(base_name).__name__}"
            )

        if sub:
            self._add_to_correct_zone_collection(zone_to_add, base_name, self._subzones)
        else:
            self._add_to_correct_zone_collection(zone_to_add, base_name, self._targets)

    def _add_to_correct_zone_collection(self, zone_to_add, base_name, loc):
        existing = loc.get(base_name)
        if existing:
            if self._zone_is_equal(existing, zone_to_add):
                return  # identical, skip adding
            else:
                # Add with counter suffix until unique
                counter = 1
                new_name = f"{base_name}_{counter}"
                while new_name in loc:
                    counter += 1
                    new_name = f"{base_name}_{counter}"
                zone_to_add.name = new_name
                loc[new_name] = zone_to_add
        else:
            loc[base_name] = zone_to_add

    def add_target(self, target_to_add: BaseTargetModel):
        """Add one target to a specific zone."""
        if isinstance(target_to_add, BaseTargetModel):
            self._targets[target_to_add.type] = target_to_add

    def add_targets(self, targets: list = []):
        """Add multiple targets to a specific zone."""
        for t in targets:
            self.add_target(t)

    def get_subzone(self, loc: str = None) -> "Zone":
        """Resolve a slash-delimited zone path relative to this zone."""
        zone = self
        if loc is None:
            return zone
        loc_address = loc.split("/", 1)
        if loc_address[0] == zone.name:
            loc_address.pop(0)
            if len(loc_address) == 0:
                return zone
            loc_address = loc_address[-1].split("/", 1)
        sub = loc_address[0]
        if sub in zone.subzones.keys():
            if len(loc_address) == 1:
                return zone.subzones[sub]
            else:
                sub_loc = loc_address[-1]
                return zone.subzones[sub].get_subzone(sub_loc)
        else:
            warnings.warn(f"Subzone '{loc}' not found.")
            return None

    def calc_utility_cost(self):
        """Calculate and cache the annual utility cost across assigned utilities."""
        self._utility_cost = sum([u.ut_cost for u in self.utility_streams])
        return self._utility_cost

    def _zone_is_equal(self, zone1: "Zone", zone2: "Zone"):
        """Basic equality check between two zones. Customize as needed."""
        return (
            zone1._hot_streams == zone2._hot_streams
            and zone1._cold_streams == zone2._cold_streams
            and zone1._hot_utilities == zone2._hot_utilities
            and zone1._cold_utilities == zone2._cold_utilities
        )

    def import_hot_and_cold_streams_from_sub_zones(
        self,
        get_net_streams: bool = False,
        is_n_zone_depth: bool = True,
        is_new_stream_collection: bool = True,
    ):
        """Get referenced hot and cold streams across multiple subzones."""
        z: Zone
        s: Stream
        if not get_net_streams:
            if is_new_stream_collection:
                self._hot_streams = StreamCollection()
                self._cold_streams = StreamCollection()
            hs_dst = self._hot_streams
            cs_dst = self._cold_streams
        else:
            if is_new_stream_collection:
                self._net_hot_streams = StreamCollection()
                self._net_cold_streams = StreamCollection()
            hs_dst = self._net_hot_streams
            cs_dst = self._net_cold_streams

        for z in self.subzones.values():
            if len(z.subzones) > 0 and is_n_zone_depth:
                z.import_hot_and_cold_streams_from_sub_zones(get_net_streams)

            if not get_net_streams:
                hs_src = z.hot_streams
                cs_src = z.cold_streams
            else:
                hs_src = z.net_hot_streams
                cs_src = z.net_cold_streams

            for s in hs_src:
                key = f"{z.name}.{s.name}"
                hs_dst.add(s, key)

            for s in cs_src:
                key = f"{z.name}.{s.name}"
                cs_dst.add(s, key)

    def get_target_zone(self, zone_name: Optional[str | list]) -> "Zone":
        """Resolve ``zone_name`` to the concrete zone that should receive a target."""
        if zone_name is None:
            return self
        resolved = str(zone_name).strip()
        if resolved == self.name:
            return self
        resolved = resolved.split("/", 1)
        if resolved[0] == self.name:
            resolved.pop(0)
        return self.get_subzone(resolved)

    def lock_dt_cont_multiplier(self):
        """Lock the dt_cont_multiplier to prevent further changes."""
        self._lock_dt_cont_multiplier = True
        self.all_streams.set_common_stream_attribute("dt_cont_multiplier_locked", True)
