"""Analysis submodules implementing OpenPinch targeting algorithms.

This package re-exports the most commonly used entry points for preparing
problem structures, running Pinch Analysis at different aggregation levels, and
assembling response payloads.  More specialised helpers remain in their
respective modules (e.g. ``support_methods`` and ``additional_analysis``).
"""

from .support_methods import *
from .graphs import visualise_graphs
from .response import output_response
from .data_preparation import prepare_problem_struture
from .problem_table_analysis import get_process_heat_cascade
from .utility_targeting import get_utility_targets
from .power_cogeneration_analysis import get_power_cogeneration_above_pinch
from .additional_analysis import *
from .area_targeting import get_area_targets
from .exchanger_unit_targeting import get_min_number_hx
