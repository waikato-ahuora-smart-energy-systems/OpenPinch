"""``design.retrofit(zone=...)`` retrofits one process of a site.

A site network is refused ("use each process's own network"), but a
process's own network on the site problem failed, since every other
process's streams had no exchanger, and ``retrofit`` had no ``zone``
(review finding 33). With ``zone`` it retrofits on that zone's own case,
as ``heat_exchanger_network(zone=...)`` designs it.
"""

from __future__ import annotations

from types import SimpleNamespace

from OpenPinch.application import case_derivation
from tests.application import test_site_heat_exchanger_network as site


def test_retrofit_with_a_zone_works_on_that_zones_own_case(monkeypatch):
    cases = []

    def zone_case(problem, zone):
        cases.append(zone.name)
        return SimpleNamespace(
            design=SimpleNamespace(retrofit=lambda existing, **kw: (existing, kw))
        )

    monkeypatch.setattr(case_derivation, "_zone_case", zone_case)
    existing = {"recovery": [{"hot": "H1", "cold": "C1", "stage": 1}]}

    network, arguments = site._site().design.retrofit(
        existing, zone="Heating", max_modifications=1
    )

    assert cases == ["Heating"]
    assert network == existing
    assert arguments["max_modifications"] == 1
