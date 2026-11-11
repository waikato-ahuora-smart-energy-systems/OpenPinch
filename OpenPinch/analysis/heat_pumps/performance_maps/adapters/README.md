# HPR Simulator Adapters

## Overview
Concrete engine backends for the engine-neutral protocols in the parent package. `coolprop.py` simulates map points with the existing `VapourCompressionCycle`; `tespy.py` builds a single-stage TESPy network for map points and for targeting candidates. The package `__init__.py` imports nothing, so TESPy is loaded only when `../factory.py` or `../targeting.py` imports `adapters.tespy` lazily.

## Interface
- `CoolPropHprPointSimulator` (`coolprop.py`): `prepare`, `simulate`, `close` session over the CoolProp cycle; returns `HprSimulatorMetadata` and `HprPointSimulation`.
- `TespyHprPointSimulator` (`tespy.py`): design solve at the nominal condition, then one offdesign solve per point restored from a private design-state file.
- `TespyHprTargetEvaluator` (`tespy.py`): implements `HprTargetEvaluator`; each candidate is a fresh design solve returning `HprTargetThermodynamicResult`.

## Invariants
- A simulator can be prepared once and closed once; a second `close` raises `HprSimulatorFailure` with `session_fatal=True`, or `HprTargetEvaluatorError` for the TESPy target evaluator, which can also be opened only once [coolprop.py:CoolPropHprPointSimulator.close, tespy.py:TespyHprPointSimulator.close, tespy.py:TespyHprTargetEvaluator.close].
- Each simulator checks that the context backend matches its own [coolprop.py:CoolPropHprPointSimulator.prepare, tespy.py:TespyHprPointSimulator.prepare].
- Map results are in kW; the CoolProp cycle takes and returns W, converted by a factor of 1000 [coolprop.py:CoolPropHprPointSimulator.simulate]. TESPy uses W internally and the same factor [tespy.py:_WATTS_PER_KILOWATT].
- Power boundary is compressor only; there are no modelled auxiliaries [../models.py:HprSimulatorMetadata].
- With a non-zero gas-side change the TESPy model adds an internal heat exchanger (TESPy `HeatExchanger`, no pressure drop): the condensed liquid heats the evaporator outlet gas, which reaches the compressor `superheat + internal_hx_gas_temperature_change` above its dew point. With zero change the network has no internal heat exchanger. [tespy.py:TespyHprPointSimulator._build_network, tespy.py:TespyHprPointSimulator._apply_condition]
- Convergence requires `network.converged` and no linear dependence [tespy.py:_network_converged].
- TESPy solve arguments come from the fixed settings object [tespy.py:TespyHprPointSimulator._solve, ../settings.py:TESPY_CONVERGENCE_SETTINGS].
- The target evaluator rejects a request whose mode, cycle, model or fluid differs from the prepared intent as session-fatal [tespy.py:TespyHprTargetEvaluator._validate_request_identity].
- The target evaluator's thermal profiles are two-point: external temperature at zero enthalpy to the saturation temperature at the duty [tespy.py:TespyHprTargetEvaluator._evaluate_fresh_design].

## Side Effects
- `TespyHprPointSimulator` creates a temporary directory and saves/restores the TESPy design state there; `close` removes it [tespy.py:TespyHprPointSimulator.prepare, tespy.py:TespyHprPointSimulator.close].
- Imports `tespy` at module import time (optional extra) [tespy.py].
- Reads package versions through `importlib.metadata` [coolprop.py:_coolprop_version, tespy.py:_tespy_version].
- Loads the packaged compressor characteristic [tespy.py:TespyHprPointSimulator._build_network].
- CoolProp saturation calls via `build_coolprop_abstract_state` [tespy.py:TespyHprPointSimulator._saturation_pressures].

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_pumps/test_hpr_coolprop_simulator.py tests/analysis/heat_pumps/test_hpr_tespy_simulator.py` (the TESPy tests need the optional `tespy` extra).
