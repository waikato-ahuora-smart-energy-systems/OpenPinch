"""An installed heat pump operated in one period (design-and-operate).

Design-and-operate for shared multiperiod designs (review finding 19,
decided 2026-11-28): each period runs the installed machine at its cheapest
operation, every compressor and exchanger needing no more than its
installed size. The capital is sunk, so a period minimises its operating
cost only.
"""

from __future__ import annotations

import dataclasses

import numpy as np

import OpenPinch.analysis.heat_pumps.targeting.cascade_vapour_compression as hp_cascade
from OpenPinch.analysis.heat_pumps._multiperiod.operation import (
    operate_installed_machine,
)
from OpenPinch.contracts.hpr import HPRSearchBudget

from .test_hpr_component_capital import _target_inputs

_SIZE_TOLERANCE = 1e-6
# A single stage's exchangers: IHX, cascade exchanger, condenser, air cooler,
# evaporator, air heater.
_CONDENSER = 2


def _fits(result, installed) -> bool:
    for machine, limit in zip(result.hpr_machine_components, installed, strict=True):
        for need, have in zip(machine.compressors, limit.compressors, strict=True):
            if need.volume_flow > have.volume_flow * (1.0 + _SIZE_TOLERANCE):
                return False
        for need, have in zip(
            machine.exchanger_areas, limit.exchanger_areas, strict=True
        ):
            if need > have * (1.0 + _SIZE_TOLERANCE):
                return False
    return True


def test_a_machine_in_its_design_period_fits_and_runs_no_dearer():
    args = _target_inputs({})
    args.refrigerant_ls = ["Ammonia"]
    args.max_multi_start = 1
    args.search_budget = HPRSearchBudget(maximum_evaluations=60)
    _, bounds = hp_cascade._get_cascade_hp_opt_setup(None, args)
    design_point = np.full(len(bounds), 0.4)
    design = hp_cascade._compute_cascade_hp_system_obj(design_point, args)

    operated = operate_installed_machine(
        design.hpr_machine_components,
        objective=hp_cascade._compute_cascade_hp_system_obj,
        initial_points=design_point,
        bounds=bounds,
        args=args,
    )

    assert _fits(operated, design.hpr_machine_components)
    assert operated.hpr_operating_cost.value <= design.hpr_operating_cost.value * (
        1.0 + _SIZE_TOLERANCE
    )


def test_a_smaller_condenser_fits_at_a_dearer_operation():
    # 70 % of the condenser: the period runs within it, at 38,951 $/y
    # against the design's 35,588 $/y.
    args = _target_inputs({})
    args.refrigerant_ls = ["Ammonia"]
    args.max_multi_start = 1
    args.search_budget = HPRSearchBudget(maximum_evaluations=200)
    _, bounds = hp_cascade._get_cascade_hp_opt_setup(None, args)
    design_point = np.full(len(bounds), 0.4)
    design = hp_cascade._compute_cascade_hp_system_obj(design_point, args)
    (machine,) = design.hpr_machine_components
    areas = list(machine.exchanger_areas)
    areas[_CONDENSER] *= 0.7
    smaller = (dataclasses.replace(machine, exchanger_areas=tuple(areas)),)

    operated = operate_installed_machine(
        smaller,
        objective=hp_cascade._compute_cascade_hp_system_obj,
        initial_points=design_point,
        bounds=bounds,
        args=args,
    )

    assert _fits(operated, smaller)
    assert operated.hpr_operating_cost.value > design.hpr_operating_cost.value


def test_a_running_compressor_moves_at_least_its_turndown():
    # A compressor ten times the design's, turned down no further than 0.9
    # of its volume flow: the period runs it at that or switches it off. It
    # ran it at 0.62 of installed without the rule.
    args = _target_inputs({})
    args.refrigerant_ls = ["Ammonia"]
    args.max_multi_start = 1
    args.search_budget = HPRSearchBudget(maximum_evaluations=200)
    args.turndown = 0.9
    _, bounds = hp_cascade._get_cascade_hp_opt_setup(None, args)
    design_point = np.full(len(bounds), 0.4)
    design = hp_cascade._compute_cascade_hp_system_obj(design_point, args)
    (machine,) = design.hpr_machine_components
    (compressor,) = machine.compressors
    installed_flow = 10.0 * compressor.volume_flow
    big = (
        dataclasses.replace(
            machine,
            compressors=(dataclasses.replace(compressor, volume_flow=installed_flow),),
        ),
    )

    operated = operate_installed_machine(
        big,
        objective=hp_cascade._compute_cascade_hp_system_obj,
        initial_points=design_point,
        bounds=bounds,
        args=args,
    )

    flow = (
        operated.hpr_machine_components[0].compressors[0].volume_flow
        if operated.hpr_machine_components
        else 0.0
    )
    assert flow <= 1e-3 * installed_flow or flow >= 0.9 * installed_flow * (
        1.0 - _SIZE_TOLERANCE
    )


def test_a_running_compressor_needs_no_more_shaft_power_than_installed():
    # The motor is sized for the compressor's peak shaft power, so a period
    # runs within it as well as within the suction volume (Codex review on
    # #566): at 70 % of the design's motor it ran the design's 0.58 kW.
    args = _target_inputs({})
    args.refrigerant_ls = ["Ammonia"]
    args.max_multi_start = 1
    args.search_budget = HPRSearchBudget(maximum_evaluations=200)
    _, bounds = hp_cascade._get_cascade_hp_opt_setup(None, args)
    design_point = np.full(len(bounds), 0.4)
    design = hp_cascade._compute_cascade_hp_system_obj(design_point, args)
    (machine,) = design.hpr_machine_components
    (compressor,) = machine.compressors
    motor = 0.7 * compressor.shaft_power
    smaller = (
        dataclasses.replace(
            machine, compressors=(dataclasses.replace(compressor, shaft_power=motor),)
        ),
    )

    operated = operate_installed_machine(
        smaller,
        objective=hp_cascade._compute_cascade_hp_system_obj,
        initial_points=design_point,
        bounds=bounds,
        args=args,
    )

    power = (
        operated.hpr_machine_components[0].compressors[0].shaft_power
        if operated.hpr_machine_components
        else 0.0
    )
    assert power <= motor * (1.0 + _SIZE_TOLERANCE)
