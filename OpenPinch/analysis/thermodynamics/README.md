# Water Thermodynamics

## Overview
Thin wrappers around CoolProp's `PropsSI` for water and steam, with bar, degC and kJ/kg at the interface and SI units inside. The steam turbine solver and the power cogeneration service in the sibling `power` package are the callers.

## Interface
- `Tsat_p(P)`, `psat_T(T)` (`water.py`): saturation temperature (degC) at a pressure (bar), and the reverse.
- `hV_p(P)`, `hL_p(P)` (`water.py`): saturated vapour and liquid enthalpy (kJ/kg) at a pressure (bar).
- `h_pT(P, T)`, `h_ps(P, s)`, `s_ph(P, H)` (`water.py`): enthalpy from pressure and temperature, enthalpy from pressure and entropy, entropy from pressure and enthalpy.
- `toSIunit_p/T/h/s`, `fromSIunit_p/T/h/s` (`water.py`): unit converters (bar/Pa, degC/K, kJ/kg to J/kg; entropy is the identity).
- `FLUID` (`water.py`): the CoolProp fluid name, `"Water"`.

## Invariants
- Pressure is in bar, temperature in degC, enthalpy in kJ/kg, and entropy in J/kg/K at every public function. [water.py:toSIunit_p] [water.py:fromSIunit_h] [water.py:toSIunit_s]
- A `None` argument to a pressure, temperature or enthalpy converter is treated as 0 before conversion. [water.py:toSIunit_T]

## Side Effects
- Imports `CoolProp.CoolProp` at module import time, so the module fails to import where CoolProp is absent. [water.py:PropsSI]
- Each property function calls CoolProp, which raises `ValueError` outside water's property range. [water.py:Tsat_p]

## Verification
- Run: `uv run --no-sync pytest tests/analysis/test_water_thermodynamics.py`
