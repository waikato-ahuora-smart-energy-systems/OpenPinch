"""Target heat pump integration for given heating or cooler profiles."""

from ..classes import *
from ..lib import *
from ..utils import *
from .problem_table_analysis import get_utility_heat_cascade, create_problem_table_with_t_int
from .gcc_manipulation import *
from .temperature_driving_force import get_temperature_driving_forces


__all__ = ["get_optimal_heat_pump_placement"]

#######################################################################################################
# Public API
#######################################################################################################


def get_optimal_heat_pump_placement(
    pt: ProblemTable,
    pt_real: ProblemTable,
    hot_utilities: StreamCollection,
    cold_utilities: StreamCollection,
    is_process_zone: bool = True,
    config: Configuration = Configuration(),
) -> Tuple[ProblemTable, ProblemTable, StreamCollection, StreamCollection]:
    """Target utility usage and compute GCC variants for a zone.

    Parameters
    ----------
    pt, pt_real:
        Shifted and real problem tables used for constructing composite curves.
    hot_utilities, cold_utilities:
        Candidate utility collections that will be targeted across temperature
        intervals.
    is_process_zone:
        When ``True`` (default) the function assumes the zone represents a
        process area and applies additional targeting logic appropriate for that
        context.

    Returns
    -------
    tuple
        Updated ``(pt, pt_real, hot_utilities, cold_utilities)`` collections with
        derived profiles embedded.
    """
    


    # Target multiple utility use




    return pt, pt_real, hot_utilities, cold_utilities


#######################################################################################################
# Helper functions
#######################################################################################################


def _get_heat_pump_heating_and_cooling_streams() -> StreamCollection:

    hp_hot_streams, hp_cold_streams = StreamCollection(), StreamCollection()

    return hp_hot_streams, hp_cold_streams


def _get_heat_pump_cascade(
    hp_hot_streams: StreamCollection, 
    hp_cold_streams: StreamCollection,
):
    pt: ProblemTable
    pt = create_problem_table_with_t_int(
        hp_hot_streams + hp_cold_streams,
        True
    )
    pt.update(
        get_utility_heat_cascade(
            pt.col[PT.T.value],
            hp_hot_streams,
            hp_cold_streams,
            is_shifted=True,
        )
    )
    pt.update(
        get_seperated_gcc_heat_load_profiles(
            pt,
            col_H_net=PT.H_UT_NET.value,
            col_H_cold_net=PT.H_COLD_UT.value,
            col_H_hot_net=PT.H_HOT_UT.value,
            is_process_stream=False,
        )
    )
    return {
        PT.T.value: pt.col[PT.T.value],
        PT.H_HOT_NET: pt.col[PT.H_HOT_NET.value],
        PT.H_COLD_NET: pt.col[PT.H_COLD_NET.value],
    }


def _get_min_temperature_approach(hp_profile) -> Tuple[float, float]:
    
    hot_side_tdf = get_temperature_driving_forces(
        T_hot=hp_profile[PT.T.value],
        H_hot=hp_profile[PT.H_HOT_NET.value],
        T_cold=None,
        H_cold=None,
    )
    min_hot_side_tdf = min(
        hot_side_tdf["delta_T1"], 
        hot_side_tdf["delta_T2"],
    )

    cold_side_tdf = get_temperature_driving_forces(
        T_hot=None,
        H_hot=None,
        T_cold=hp_profile[PT.T.value],
        H_cold=hp_profile[PT.H_HOT_NET.value],
    )
    min_cold_side_tdf = min(
        cold_side_tdf["delta_T1"], 
        cold_side_tdf["delta_T2"],
    )
    return min_hot_side_tdf, min_cold_side_tdf
