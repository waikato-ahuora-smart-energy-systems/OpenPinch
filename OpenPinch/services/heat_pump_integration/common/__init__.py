"""Common helpers shared across heat-pump targeting models."""

from . import encoding, preprocessing, shared
from .encoding import (
    map_DT_arr_to_x_arr,
    map_Q_amb_to_x,
    map_Q_arr_to_x_arr,
    map_T_arr_to_x_arr,
    map_x_arr_to_DT_arr,
    map_x_arr_to_Q_arr,
    map_x_arr_to_T_arr,
    map_x_to_Q_amb,
)
from .preprocessing import construct_HPRTargetInputs
from .shared import (
    calc_carnot_heat_engine_eta,
    calc_carnot_heat_pump_cop,
    calc_hpr_obj,
    compute_entropic_mean_temperature,
    create_stream_collection_of_background_profile,
    get_ambient_air_stream,
    get_Q_vals_at_T_hpr_from_bckgrd_profile,
    get_carnot_hpr_cycle_streams,
    plot_multi_hp_profiles_from_results,
    solve_hpr_placement,
    validate_vapour_hp_refrigerant_ls,
)

__all__ = [
    "calc_carnot_heat_engine_eta",
    "calc_carnot_heat_pump_cop",
    "calc_hpr_obj",
    "compute_entropic_mean_temperature",
    "construct_HPRTargetInputs",
    "create_stream_collection_of_background_profile",
    "encoding",
    "get_Q_vals_at_T_hpr_from_bckgrd_profile",
    "get_ambient_air_stream",
    "get_carnot_hpr_cycle_streams",
    "map_DT_arr_to_x_arr",
    "map_Q_amb_to_x",
    "map_Q_arr_to_x_arr",
    "map_T_arr_to_x_arr",
    "map_x_arr_to_DT_arr",
    "map_x_arr_to_Q_arr",
    "map_x_arr_to_T_arr",
    "map_x_to_Q_amb",
    "plot_multi_hp_profiles_from_results",
    "preprocessing",
    "shared",
    "solve_hpr_placement",
    "validate_vapour_hp_refrigerant_ls",
]
