"""A retrofit design's existing units keep a valid area.

A unit rated within the area tolerance may need slightly more area by its
slices than is installed: on the Hallberg and Scribner crude case E9 needs
1,222.6 m2 against 1,211.2 m2 installed. An exchanger whose area is below
its slices' is rejected when a network is loaded, so a design that reported
the installed area there could not be read back from its own dump.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.domain.heat_exchanger import HeatExchanger
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from tests.domain.test_heat_exchanger_area_slices import _exchanger, _slice


def _network(slices_area: float) -> HeatExchangerNetwork:
    unit = _exchanger(slices=(_slice(period="0", duty=100.0, area=slices_area),))
    return HeatExchangerNetwork(exchangers=(unit,))


def _owned(slices_area: float, installed: float) -> HeatExchanger:
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        _with_owned_areas,
    )

    network = _network(slices_area)
    (unit,) = network.exchangers
    owned = _with_owned_areas(
        network,
        {unit.exchanger_id: SimpleNamespace(area=installed)},
        {unit.exchanger_id: installed},
    )
    # The design reads back from its own dump.
    assert HeatExchangerNetwork.model_validate(owned.model_dump()) == owned
    return owned.exchangers[0]


def test_a_unit_with_spare_area_reports_what_is_installed() -> None:
    assert _owned(slices_area=80.0, installed=100.0).area == pytest.approx(100.0)


def test_a_unit_whose_slices_need_more_than_is_installed_reports_that() -> None:
    assert _owned(slices_area=104.0, installed=100.0).area == pytest.approx(104.0)
