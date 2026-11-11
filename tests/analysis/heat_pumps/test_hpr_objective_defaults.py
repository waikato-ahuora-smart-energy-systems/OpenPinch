"""Objective defaults for heat pump and refrigeration targets."""

from __future__ import annotations

import numpy as np

from OpenPinch.analysis.heat_pumps.common import preprocessing as hp_pre
from OpenPinch.analysis.heat_pumps.common.shared import heating_allowance
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.configuration_fields import CONFIG_FIELD_SPECS


def test_displaced_utility_capital_is_not_credited_by_default():
    # A retrofit keeps its boiler and refrigeration plant, so a heat pump that
    # displaces their duty does not save their capital.
    assert (
        CONFIG_FIELD_SPECS["COSTING_HPR_UTILITY_CAPITAL_RECOVERY_ENABLED"].default
        is False
    )


def _target_inputs(options):
    return hp_pre.construct_HPRTargetInputs(
        Q_hpr_target=50.0,
        T_vals=np.array([120.0, 80.0, 40.0]),
        H_hot=np.array([0.0, -50.0, -100.0]),
        H_cold=np.array([100.0, 50.0, 0.0]),
        config=Configuration(options=options),
    )


def test_a_heat_pump_meets_its_heating_duty_only_when_asked():
    assert _target_inputs({}).enforce_heating_duty is False
    assert (
        _target_inputs({"HPR_ENFORCE_HEATING_DUTY": True}).enforce_heating_duty is True
    )


def test_the_heating_allowance_is_computed_once_on_the_target_inputs():
    args = _target_inputs({"HPR_ENFORCE_HEATING_DUTY": True})

    allowance = heating_allowance(args)

    assert args.heating_allowance == allowance
