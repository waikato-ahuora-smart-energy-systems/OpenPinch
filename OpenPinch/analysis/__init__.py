"""Analysis submodules implementing OpenPinch targeting algorithms.

This package re-exports the most commonly used entry points for preparing
problem structures, running Pinch Analysis at different aggregation levels, and
assembling response payloads.  More specialised helpers remain in their
respective modules (e.g. ``support_methods`` and ``additional_analysis``).
"""

from .graphs import visualise_graphs
from .response import output_response
from .data_preparation import prepare_problem_struture
from .operation_analysis import get_energy_transfer_retrofit_analysis
from .problem_table_analysis import problem_table_algorithm, calc_problem_table
from .process_analysis import get_process_pinch_targets
from .power_cogeneration_analysis import get_power_cogeneration_above_pinch
from .site_analysis import get_site_targets
from .region_analysis import get_regional_targets
from .utility_targeting import get_zonal_utility_targets, target_utility, calc_GGC_utility
from .support_methods import *
from .additional_analysis import *
