import pytest
import matplotlib.pyplot as plt
import numpy as np
from OpenPinch.analysis.stream_linearisation import *

# TODO 

def test_build_curve_pure():
    supply_temp = 300 
    target_temp = 500
    composition = [("water", 1)]
    num_points = 50
    pressure = 200*1000 #kPa
    # points = generate_t_h_curve(ppKey="", composition=composition, mole_flow=1, t_supply=supply_temp, t_target=target_temp, p_supply=pressure, p_target=pressure, num_points=num_points)
    # # Validate number of points (No missing values)
    # assertEqual(num_points, points[0].size/points[0].ndim, 'Missing temperature/enthalpy value/s')

    # # Validate temperature range
    # assertEqual(supply_temp, points[0][0][1], f'Temperature interval range is inaccurate, range starts from {points[0][0][1]} instead of {supply_temp}')
    # assertEqual(target_temp, points[0][-1][1], f'Temperature interval range is inaccurate, range ends at {points[0][-1][1]} instead of {target_temp}')

# def test_build_curve_mixture():
#     supply_temp = 300 
#     target_temp = 750
#     composition = [("water", 0.5), ("ethanol", 0.5)]
#     num_points = 50
#     pressure = 101325

#     points = generate_t_h_curve(ppKey="", composition=composition, mole_flow=1, t_supply=supply_temp, t_target=target_temp, p_supply=pressure, p_target=pressure,  num_points=num_points)

#     # Validate number of points (No missing values)
#     assertEqual(num_points, points[0].size/points[0].ndim, 'Missing temperature/enthalpy value/s')

#     # Validate temperature range
#     assertEqual(supply_temp, points[0][0][1], f'Temperature interval range is inaccurate, range starts from {points[0][0][1]} instead of {supply_temp}')
#     assertEqual(target_temp, points[0][-1][1], f'Temperature interval range is inaccurate, range ends at {points[0][-1][1]} instead of {target_temp}')

# def test_piecewise_curve():
#     supply_temp = 300 
#     target_temp = 750
#     composition = [("water", 0.5), ("ethanol", 0.5)]
#     num_points = 50
#     pressure = 101325

#     points = generate_t_h_curve(ppKey="", composition=composition, mole_flow=1, t_supply=supply_temp, t_target=target_temp, p_supply=pressure, p_target=pressure,  num_points=num_points)
#     n_points = pw_curve(points[0], 1, False)

#     assertGreater(points[0].size/points[0].ndim, len(n_points), "Curve was not simplified, number of points is equal")
#     assertEqual(points[0].all(), n_points[0].all(), "Simplified curve has changed the starting structure of the curve")
#     assertEqual(points[0].all(), n_points[-1].all(), "Simplified curve has changed the final structure of the curve")

# def test_piecewise_curve_small_epsilon():
#     supply_temp = 300 
#     target_temp = 750
#     composition = [("water", 0.5), ("ethanol", 0.5)]
#     num_points = 50
#     pressure = 101325

#     points = generate_t_h_curve(ppKey="", composition=composition, mole_flow=1, t_supply=supply_temp, t_target=target_temp, p_supply=pressure, p_target=pressure,  num_points=num_points)
#     n_points = pw_curve(points[0], 1e-5, False) # Tiny tolerance - expect curves to be the same

#     assertEqual(points[0].size/points[0].ndim, len(n_points), "Curve was simplified when it should not be, number of points is not equal")
#     assertEqual(points[0].all(), n_points[0].all(), "Simplified curve has changed the starting structure of the curve")
#     assertEqual(points[0].all(), n_points[-1].all(), "Simplified curve has changed the final structure of the curve")
