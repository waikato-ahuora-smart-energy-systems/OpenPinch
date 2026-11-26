"""The retrofit's heat pump step targets at the existing network's multiplier.

A retrofit of a network designed at a dt_cont multiplier holds every
process stream to its contribution at that multiplier. The as-piped heat
pump target is solved on the existing coolers and heaters shifted by the
same contributions, so the machine's levels stand where the retrofit's
model can serve them: at 1.5, each cooler and heater is shifted by 1.5
times its stream's contribution.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.retrofit import heat_pump_targets
from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
    RetrofitSettings,
)
from OpenPinch.application.heat_pump_retrofit import heat_pump_step
from tests.analysis.heat_exchanger_networks.test_retrofit_heat_pump_targets import (
    DT_CONT,
    KELVIN,
    _problem,
    _solved_network,
)


def test_the_step_targets_on_profiles_shifted_at_the_networks_multiplier(
    monkeypatch,
):
    profiles = []

    def target(problem, profile, prices, args, **arguments):
        profiles.append(profile)
        return None

    monkeypatch.setattr(heat_pump_targets, "_profile_target", target)
    settings = RetrofitSettings(
        heat_pump_cycle="carnot", approach=None, dt_cont_multiplier=1.5
    )

    heat_pump_step(
        _problem(), SimpleNamespace(network=_solved_network()), settings, machine="HP"
    )

    # Shifted temperatures (degC), hottest first: the heater's outlet raised
    # and the coldest cooler outlet lowered by 1.5 times the contribution.
    T_vals, _H_hot, _H_cold = profiles[0]
    assert T_vals[0] == pytest.approx(538.15 - KELVIN + 1.5 * DT_CONT)
    assert T_vals[-1] == pytest.approx(350.15 - KELVIN - 1.5 * DT_CONT)
