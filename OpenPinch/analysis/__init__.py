"""Analysis submodules implementing OpenPinch targeting algorithms.

This package re-exports the most commonly used entry points for preparing
problem structures, running Pinch Analysis at different aggregation levels, and
assembling response payloads.  More specialised helpers remain in their
respective modules (e.g. ``support_methods`` and ``additional_analysis``).
"""

from .additional_analysis import *
from .area_targeting import get_area_targets
from .data_preparation import prepare_problem
from .exchanger_unit_targeting import get_min_number_hx
from .graph_data import visualise_graphs
from .power_cogeneration_analysis import get_power_cogeneration_above_pinch
from .problem_table_analysis import get_process_heat_cascade
from .support_methods import *
from .utility_targeting import get_utility_targets
