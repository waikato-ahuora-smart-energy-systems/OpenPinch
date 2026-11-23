"""A one-sided problem honours the forbidden matches by refusing.

A problem with only cold (or only hot) streams has one network: heaters (or
coolers) at the utility levels targeting gives. Nothing can be traded, so a
forbidden pair that network needs cannot be designed around; the run says so
rather than return it.
"""

from __future__ import annotations

import pytest

from tests.analysis.heat_exchanger_networks.test_utility_only_design import _cold_only


def _pairs(network) -> set[tuple[str, str]]:
    return {
        (e.source_stream.rsplit(".", 1)[-1], e.sink_stream.rsplit(".", 1)[-1])
        for e in network.exchangers
    }


def test_a_utility_only_design_that_needs_a_forbidden_pair_is_refused():
    used = _pairs(_cold_only().design.heat_exchanger_network().selected_network)
    utility, stream = sorted(used)[0]

    with pytest.raises(ValueError, match="HENS_FORBIDDEN_MATCHES") as refused:
        _cold_only(
            options={"HENS_FORBIDDEN_MATCHES": [[utility, stream]]}
        ).design.heat_exchanger_network()
    assert utility in str(refused.value) and stream in str(refused.value)

    # A pair the network does not use changes nothing.
    unused = next(
        (u, s)
        for u in ("MPS", "LPS", "HW")
        for s in sorted({cold for _hot, cold in used})
        if (u, s) not in used
    )
    spared = _cold_only(
        options={"HENS_FORBIDDEN_MATCHES": [list(unused)]}
    ).design.heat_exchanger_network()
    assert _pairs(spared.selected_network) == used
