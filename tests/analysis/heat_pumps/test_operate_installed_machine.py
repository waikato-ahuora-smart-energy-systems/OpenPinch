"""An installed heat pump operated in one period (design-and-operate).

Design-and-operate for shared multiperiod designs (review finding 19,
decided 2026-11-28): each period runs the installed machine at its cheapest
operation, every compressor and exchanger needing no more than its
installed size. The capital is sunk, so a period minimises its operating
cost only.

The tests check how candidates are ranked and accepted, not how far the
search goes: each starts from the design point and the same point at a
tenth of its heat (which fits any of these machines) and evaluates at most
six points, so the file runs in seconds under Pyodide too (one evaluation
takes about half a second there; at 200 evaluations the file took 7.5 min
and pushed the heat pump shard past its 20 min limit).
"""

from __future__ import annotations

import dataclasses

import numpy as np

import OpenPinch.analysis.heat_pumps.targeting.cascade_vapour_compression as hp_cascade
from OpenPinch.analysis.heat_pumps._multiperiod.operation import (
    operate_installed_machine,
)
from OpenPinch.analysis.heat_pumps.common.layout import HPRoptVectorLayout
from OpenPinch.contracts.hpr import HPRSearchBudget
from OpenPinch.domain.enums import HeatPumpAndRefrigerationCycle

from .test_hpr_component_capital import _target_inputs

_SIZE_TOLERANCE = 1e-6
# A single stage's exchangers: IHX, cascade exchanger, condenser, air cooler,
# evaporator, air heater.
_CONDENSER = 2
# Enough for the search to reach its second start.
_SMALL_SEARCH = 6
_LAYOUT = HPRoptVectorLayout(
    n_cond=1, n_evap=1, n_subcool=1, n_heat_split=1, n_cool_split=0, n_ihx=1
)


def _case(*, evaluations: int, **options):
    """A single-stage ammonia heat pump's period, its design point (0.4 on
    every axis) and the design's result; the search evaluates at most
    ``evaluations`` points."""
    args = _target_inputs(options)
    args.refrigerant_ls = ["Ammonia"]
    args.max_multi_start = 1
    args.search_budget = HPRSearchBudget(maximum_evaluations=evaluations)
    _, bounds = hp_cascade._get_cascade_hp_opt_setup(None, args)
    design_point = np.full(len(bounds), 0.4)
    design = hp_cascade._compute_cascade_hp_system_obj(design_point, args)
    return args, bounds, design_point, design


def _with_low_duty(design_point: np.ndarray) -> np.ndarray:
    """The design point, then the same point at a tenth of its heat."""
    low = design_point.copy()
    low[_LAYOUT.heat_split_slice] *= 0.1
    return np.vstack([design_point, low])


def _operate(installed, args, bounds, starts):
    return operate_installed_machine(
        installed,
        objective=hp_cascade._compute_cascade_hp_system_obj,
        initial_points=starts,
        bounds=bounds,
        args=args,
    )


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
    args, bounds, design_point, design = _case(evaluations=1)

    operated = _operate(design.hpr_machine_components, args, bounds, design_point)

    assert _fits(operated, design.hpr_machine_components)
    assert operated.hpr_operating_cost.value <= design.hpr_operating_cost.value * (
        1.0 + _SIZE_TOLERANCE
    )


def test_a_smaller_condenser_fits_at_a_dearer_operation():
    # 70 % of the condenser: the design point does not fit it, the low-duty
    # point does, at a higher operating cost.
    args, bounds, design_point, design = _case(evaluations=_SMALL_SEARCH)
    (machine,) = design.hpr_machine_components
    areas = list(machine.exchanger_areas)
    areas[_CONDENSER] *= 0.7
    smaller = (dataclasses.replace(machine, exchanger_areas=tuple(areas)),)

    operated = _operate(smaller, args, bounds, _with_low_duty(design_point))

    assert _fits(operated, smaller)
    assert operated.hpr_operating_cost.value > design.hpr_operating_cost.value


def test_a_running_compressor_moves_at_least_its_turndown():
    # A compressor ten times the design's, turned down no further than 0.9
    # of its volume flow (HPR_TURNDOWN_DUTY): the period runs it at that or
    # switches it off. It ran it at 0.62 of installed without the rule.
    args, bounds, design_point, design = _case(
        evaluations=_SMALL_SEARCH, HPR_TURNDOWN_DUTY=0.9
    )
    (machine,) = design.hpr_machine_components
    (compressor,) = machine.compressors
    installed_flow = 10.0 * compressor.volume_flow
    big = (
        dataclasses.replace(
            machine,
            compressors=(dataclasses.replace(compressor, volume_flow=installed_flow),),
        ),
    )

    operated = _operate(big, args, bounds, _with_low_duty(design_point))

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
    args, bounds, design_point, design = _case(evaluations=_SMALL_SEARCH)
    (machine,) = design.hpr_machine_components
    (compressor,) = machine.compressors
    motor = 0.7 * compressor.shaft_power
    smaller = (
        dataclasses.replace(
            machine, compressors=(dataclasses.replace(compressor, shaft_power=motor),)
        ),
    )

    operated = _operate(smaller, args, bounds, _with_low_duty(design_point))

    power = (
        operated.hpr_machine_components[0].compressors[0].shaft_power
        if operated.hpr_machine_components
        else 0.0
    )
    assert power <= motor * (1.0 + _SIZE_TOLERANCE)


def test_an_operation_that_does_not_fit_is_no_operation():
    # One evaluation (the design point) on half the design's condenser: the
    # search's best still needs twice the area, so the period has no
    # operation rather than one that breaks the machine (Codex review on
    # #566).
    args, bounds, design_point, design = _case(evaluations=1)
    (machine,) = design.hpr_machine_components
    areas = list(machine.exchanger_areas)
    areas[_CONDENSER] *= 0.5
    smaller = (dataclasses.replace(machine, exchanger_areas=tuple(areas)),)

    operated = _operate(smaller, args, bounds, design_point)

    assert not operated.success


def test_a_running_stage_holds_its_pressure_ratio_turndown():
    # A compressor designed for five times the design point's pressure
    # ratio, turned down no further than 0.9 of it
    # (HPR_TURNDOWN_PRESSURE_RATIO): the period runs it at that or not at
    # all. Without the rule it ran at the design point's ratio, 1.39.
    args, bounds, design_point, design = _case(
        evaluations=_SMALL_SEARCH, HPR_TURNDOWN_PRESSURE_RATIO=0.9
    )
    (machine,) = design.hpr_machine_components
    (compressor,) = machine.compressors
    design_ratio = 5.0 * compressor.pressure_ratio
    lifted = (
        dataclasses.replace(
            machine,
            compressors=(dataclasses.replace(compressor, pressure_ratio=design_ratio),),
        ),
    )

    operated = _operate(lifted, args, bounds, _with_low_duty(design_point))

    ratio = (
        operated.hpr_machine_components[0].compressors[0].pressure_ratio
        if operated.hpr_machine_components
        else 0.0
    )
    assert ratio == 0.0 or ratio >= 0.9 * design_ratio * (1.0 - _SIZE_TOLERANCE)


def test_the_best_operation_that_fits_is_taken():
    # A condenser 0.1 % short of the design's: the design point ranks first
    # on its small penalty but does not fit, the low-duty point does. The
    # period was left without the machine, as the turndown period of the
    # reviewer's three-period case was though its own design fitted.
    args, bounds, design_point, design = _case(evaluations=_SMALL_SEARCH)
    (machine,) = design.hpr_machine_components
    areas = list(machine.exchanger_areas)
    areas[_CONDENSER] *= 0.999
    smaller = (dataclasses.replace(machine, exchanger_areas=tuple(areas)),)

    operated = _operate(smaller, args, bounds, _with_low_duty(design_point))

    assert operated.success
    assert _fits(operated, smaller)


def _always_penalised(x, args, **kwargs):
    """The cycle objective with every solved point outside a cycle limit."""
    result = hp_cascade._compute_cascade_hp_system_obj(x, args, **kwargs)
    if not result.success:
        return result
    return result.with_updates(feasibility_penalty=1e6, obj=result.obj + 1e6)


def test_an_operation_outside_the_cycle_limits_is_no_operation():
    # Every candidate breaks a cycle limit: the period has no operation. The
    # VC+MVR crude preheat target took such operations and was then refused
    # whole for their penalty.
    args, bounds, design_point, design = _case(evaluations=_SMALL_SEARCH)
    # Carnot designs are exempt from the penalty check; these are cycles.
    args.hpr_type = HeatPumpAndRefrigerationCycle.CascadeVapourComp.value

    operated = operate_installed_machine(
        design.hpr_machine_components,
        objective=_always_penalised,
        initial_points=_with_low_duty(design_point),
        bounds=bounds,
        args=args,
    )

    assert not operated.success
