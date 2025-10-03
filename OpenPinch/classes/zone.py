"""Zone data structure capturing nested scopes and their thermal targets."""

from typing import Optional, TYPE_CHECKING
from ..lib.enums import *
from ..lib.config import *
from .stream_collection import StreamCollection
from .target import EnergyTarget

if TYPE_CHECKING:
    from .stream import Stream
    from .zone import Zone


class Zone():
    """Class representing any type of defined spatial zone (i.e., operation, 
    process zone, site, region, utility zone) with its energy targets.
    """

    def __init__(self, name: str = "Zone", identifier: str = ZoneType.P.value, config: Optional[Configuration] = None, parent_zone: "Zone" = None):
        
        # === Metadata ===
        self._name = name
        self._identifier = identifier
        self._config = config or Configuration()
        self._parent_zone = parent_zone
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
    def name(self): return self._name
    @name.setter
    def name(self, value): self._name = value

    @property
    def identifier(self): return self._identifier
    @identifier.setter
    def identifier(self, value): self._identifier = value

    @property
    def config(self): return self._config
    @config.setter
    def config(self, value): self._config = value

    @property
    def parent_zone(self): return self._parent_zone
    @parent_zone.setter
    def parent_zone(self, value): self._parent_zone = value    

    @property
    def active(self) -> bool: return bool(self._active)
    @active.setter
    def active(self, value: bool): self._active = bool(value)

    @property
    def hot_streams(self): return self._hot_streams
    @hot_streams.setter
    def hot_streams(self, data): self._hot_streams = data

    @property
    def cold_streams(self): return self._cold_streams
    @cold_streams.setter
    def cold_streams(self, data): self._cold_streams = data 

    @property
    def net_hot_streams(self): return self._net_hot_streams
    @net_hot_streams.setter
    def net_hot_streams(self, data): self._net_hot_streams = data

    @property
    def net_cold_streams(self): return self._net_cold_streams
    @net_cold_streams.setter
    def net_cold_streams(self, data): self._net_cold_streams = data           

    @property
    def hot_utilities(self): return self._hot_utilities
    @hot_utilities.setter
    def hot_utilities(self, data): self._hot_utilities = data

    @property
    def cold_utilities(self): return self._cold_utilities
    @cold_utilities.setter
    def cold_utilities(self, data): self._cold_utilities = data    

    @property
    def graphs(self): return self._graphs
    @graphs.setter
    def graphs(self, data): self._graphs = data  

    @property
    def subzones(self): return self._subzones

    @property
    def targets(self): return self._targets    
    
    @property
    def process_streams(self):
        return self._hot_streams + self._cold_streams
    
    @property
    def net_process_streams(self):
        return self._net_hot_streams + self._net_cold_streams

    @property
    def utility_streams(self):
        return self._hot_utilities + self._cold_utilities 

    @property
    def all_streams(self):
        return self.process_streams + self.utility_streams

    @property
    def all_net_streams(self):
        return self.net_process_streams + self.utility_streams
    

    # === Methods ===
    def add_graph(self, name: str, result):
        self._graphs[name] = result


    def add_zone(self, zone_to_add, sub: bool =True):
        """Add a single zone object keyed by its name.

        If the zone name already exists:
        - If the zone is identical (e.g. same stream and utility objects), skip.
        - If it's different, add it with a suffix like '_1', '_2', etc.
        """
        base_name = getattr(zone_to_add, "name", None)

        if not isinstance(base_name, str):
            raise ValueError(f"Zone must have a string 'name' attribute, got: {type(base_name).__name__}")

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


    def add_target(self, target_to_add: EnergyTarget):
        """Add one target to a specific zone."""
        self._targets[target_to_add.name] = target_to_add


    def add_targets(self, targets: list):
        """Add multiple targets to a specific zone."""
        for t in targets:
            self.add_target(t)


    def add_target_from_results(self, target_id: str = None, results: dict = None):
        target_name = f"{self.name}/{target_id}" if target_id is not None else self.name
        res = EnergyTarget(target_name, target_id, self.parent_zone, config=self.config)
        for key, value in results.items():
            setattr(res, key, value)
        self.add_target(res)


    def get_subzone(self, loc: str):
        loc_address = loc.split("/")
        zone = self
        for sub in loc_address:
            try:
                zone = zone.subzones[sub]
            except KeyError as exc:
                raise ValueError(f"Subzone '{loc}' not found.") from exc
        return zone


    def calc_utility_cost(self):
        self._utility_cost = sum([u.ut_cost for u in self.utility_streams])
        return self._utility_cost


    def _zone_is_equal(self, zone1: "Zone", zone2: "Zone"):
        """Basic equality check between two zones. Customize as needed."""
        return (
            zone1._hot_streams == zone2._hot_streams and
            zone1._cold_streams == zone2._cold_streams and
            zone1._hot_utilities == zone2._hot_utilities and
            zone1._cold_utilities == zone2._cold_utilities
        )


    def import_hot_and_cold_streams_from_sub_zones(self):
        """Get hot and cold streams from multiple subzones into two separate lists, maintaining references."""
        z: Zone
        s: Stream
        for z in self.subzones.values():
            if len(z.subzones) > 0:
                z.import_hot_and_cold_streams_from_sub_zones()
    
            for s in z.hot_streams: 
                key = f"{z.name}.{s.name}"
                self._hot_streams.add(s, key)
                
            for s in z.cold_streams: 
                key = f"{z.name}.{s.name}"
                self._cold_streams.add(s, key)


    def import_net_hot_and_cold_streams_from_sub_zones(self):
        """Get net hot and cold streams from multiple subzones into two separate lists, maintaining references."""
        z: Zone
        s: Stream
        for z in self.subzones.values():
            if len(z.subzones) > 0:
                z.import_hot_and_cold_streams_from_sub_zones()            

            for s in z.net_hot_streams: 
                key = f"{z.name}.{s.name}"
                self._net_hot_streams.add(s, key)
                
            for s in z.net_cold_streams: 
                key = f"{z.name}.{s.name}"
                self._net_cold_streams.add(s, key)
