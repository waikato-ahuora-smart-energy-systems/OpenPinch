from typing import TYPE_CHECKING
import pandas as pd
from ..lib.schema import *
from ..lib.enums import *
from ..lib.config import *
from .value import Value
from .stream_collection import StreamCollection
from .problem_table import ProblemTable

if TYPE_CHECKING:
    from .zone import Zone

class Target():
    """Class representing energy targets."""

    def __init__(self, name: str = "untitled", identifier: str = TargetType.DI.value, parent_zone: "Zone" = None, config: Optional[Configuration] = None):

        # === Metadata ===
        self._config = config or Configuration()
        self._parent_zone = parent_zone
        self._identifier = identifier
        self._name = name
        self._active = True

        self._graphs = {}
        
        # === Streams & Utilities ===
        self._hot_utilities: StreamCollection = StreamCollection()
        self._cold_utilities: StreamCollection = StreamCollection()        

        # === System Data ===
        self._problem_table = ProblemTable({col.value: pd.Series(dtype='float64') for col in ProblemTableLabel})
        self._problem_table_real = ProblemTable({col.value: pd.Series(dtype='float64') for col in ProblemTableLabel})

        # === Thermal Targeting ===
        self._hot_pinch = 0.0
        self._cold_pinch = 0.0
        self._heat_recovery_target = 0.0
        self._heat_recovery_limit = 0.0
        self._hot_utility_target = 0.0
        self._cold_utility_target = 0.0
        self._utility_heat_recovery_target = 0.0
        self._degree_of_int = 100.0
        self._num_units = 0

        # === Exergy Targeting ===
        self._exergy_sinks = 0.0
        self._exergy_sources = 0.0
        self._exergy_des_min = 0.0
        self._exergy_req_min = 0.0
        self._w_target = 0.0
        self._w_eff_target = 0.0
        self._ETE = 0.0

        # === Cost Targeting ===
        self._area = 0.0
        self._capital_cost = 0.0
        self._total_cost = 0.0
        self._utility_cost = 0.0

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
    def active(self) -> bool:
        """Whether the stream is active in analysis."""
        if isinstance(self._active, Value):
            return self._active.value
        else:
            return self._active
    @active.setter
    def active(self, value: bool):
        self._active = Value(value)

    @property
    def pt(self): return self._problem_table
    @pt.setter
    def pt(self, data): self._problem_table = data

    @property
    def pt_real(self): return self._problem_table_real
    @pt_real.setter
    def pt_real(self, data): self._problem_table_real = data

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
    def utility_streams(self):
        return self._hot_utilities + self._cold_utilities 


    @property
    def area(self): return self._area
    @area.setter
    def area(self, value): self._area = value

    @property
    def capital_cost(self): return self._capital_cost
    @capital_cost.setter
    def capital_cost(self, value): self._capital_cost = value

    @property
    def cold_pinch(self): return self._cold_pinch
    @cold_pinch.setter
    def cold_pinch(self, value): self._cold_pinch = value

    @property
    def cold_utility_target(self): return self._cold_utility_target
    @cold_utility_target.setter
    def cold_utility_target(self, value): self._cold_utility_target = value

    @property
    def degree_of_int(self): return self._degree_of_int
    @degree_of_int.setter
    def degree_of_int(self, value): self._degree_of_int = value

    @property
    def ETE(self): return self._ETE
    @ETE.setter
    def ETE(self, value): self._ETE = value

    @property
    def exergy_des_min(self): return self._exergy_des_min
    @exergy_des_min.setter
    def exergy_des_min(self, value): self._exergy_des_min = value

    @property
    def exergy_req_min(self): return self._exergy_req_min
    @exergy_req_min.setter
    def exergy_req_min(self, value): self._exergy_req_min = value

    @property
    def exergy_sinks(self): return self._exergy_sinks
    @exergy_sinks.setter
    def exergy_sinks(self, value): self._exergy_sinks = value

    @property
    def exergy_sources(self): return self._exergy_sources
    @exergy_sources.setter
    def exergy_sources(self, value): self._exergy_sources = value

    @property
    def heat_recovery_target(self): return self._heat_recovery_target
    @heat_recovery_target.setter
    def heat_recovery_target(self, value): self._heat_recovery_target = value

    @property
    def heat_recovery_limit(self): return self._heat_recovery_limit
    @heat_recovery_limit.setter
    def heat_recovery_limit(self, value): self._heat_recovery_limit = value

    @property
    def utility_heat_recovery_target(self): return self._utility_heat_recovery_target
    @utility_heat_recovery_target.setter
    def utility_heat_recovery_target(self, value): self._utility_heat_recovery_target = value

    @property
    def hot_pinch(self): return self._hot_pinch
    @hot_pinch.setter
    def hot_pinch(self, value): self._hot_pinch = value

    @property
    def hot_utility_target(self): return self._hot_utility_target
    @hot_utility_target.setter
    def hot_utility_target(self, value): self._hot_utility_target = value

    @property
    def num_units(self): return self._num_units
    @num_units.setter
    def num_units(self, value): self._num_units = value

    @property
    def capital_cost(self): return self._capital_cost
    @capital_cost.setter
    def capital_cost(self, value): self._capital_cost = value

    @property
    def total_cost(self): return self._total_cost
    @total_cost.setter
    def total_cost(self, value): self._total_cost = value

    @property
    def utility_cost(self): return self._utility_cost
    @utility_cost.setter
    def utility_cost(self, value): self._utility_cost = value

    @property
    def work_target(self): return self._w_target
    @work_target.setter
    def work_target(self, value): self._w_target = value

    @property
    def turbine_efficiency_target(self): return self._w_eff_target
    @turbine_efficiency_target.setter
    def turbine_efficiency_target(self, value): self._w_eff_target = value

    @property
    def target_values(self): return self._target_values
    @target_values.setter
    def target_values(self, value_dict): 
        self._target_values = value_dict
        for key, value in value_dict.items():
            setattr(self, key, value)        


    # === Methods ===
    def add_graph(self, name: str, result):
        self._graphs[name] = result


    def calc_utility_cost(self):
        self._utility_cost = sum([u.ut_cost for u in self.utility_streams])
        return self._utility_cost


    def serialize_json(self, isTotal=False):
        """
        Serialize process into json for return data
        """
        
        degree_of_integration = None
        if(self.degree_of_int):
            degree_of_integration = self.degree_of_int * 100

        data = {
            'name': self.name,
            'degree_of_integration': degree_of_integration,
            'Qh': self.hot_utility_target,
            'Qc': self.cold_utility_target,
            'Qr': self.heat_recovery_target,
            'utility_cost': self.utility_cost,
            'row_type': SummaryRowType.FOOTER.value if isTotal else SummaryRowType.CONTENT.value,
            'hot_utilities': [],
            'cold_utilities': [],
            'temp_pinch': {
                'cold_temp': None,
                'hot_temp': None
            }
        }

        if isinstance(self.cold_pinch, float) and isinstance(self.hot_pinch, float):
            if abs(self.cold_pinch - self.hot_pinch) < ZERO:
                temp_pinch = {'cold_temp': self.cold_pinch}
                data['temp_pinch'] = temp_pinch
            else:
                temp_pinch = {'cold_temp': self.cold_pinch, 'hot_temp': self.hot_pinch}
                data['temp_pinch'] = temp_pinch
        elif isinstance(self.cold_pinch, float):
            temp_pinch = {'cold_temp': self.cold_pinch}
            data['temp_pinch'] = temp_pinch            
        elif isinstance(self.hot_pinch, float):
            temp_pinch = {'hot_temp': self.hot_pinch}
            data['temp_pinch'] = temp_pinch   

        if self.config.TURBINE_WORK_BUTTON:
            data['work_target'] = self.work_target
            data['turbine_efficiency_target'] = self.turbine_efficiency_target * 100

        if self.config.AREA_BUTTON:
            data['area'] = self.area
            data['num_units'] = self.num_units
            data['capital_cost'] = self.capital_cost
            data['total_cost'] = self.total_cost
        
        if self.config.EXERGY_BUTTON:
            data['exergy_sources'] = self.exergy_sources
            data['exergy_sinks'] = self.exergy_sinks
            data['ETE'] = self.ETE * 100
            data['exergy_req_min'] = self.exergy_req_min
            data['exergy_des_min'] = self.exergy_des_min

        for utility in self.hot_utilities:
            data['hot_utilities'].append({'name': utility.name, 'heat_flow': utility.heat_flow})
        for utility in self.cold_utilities:
            data['cold_utilities'].append({'name': utility.name, 'heat_flow': utility.heat_flow})
        
        return data
