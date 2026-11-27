"""The median design basis says which streams it leaves out.

A stream that runs under ``HENS_BASIS_ACTIVE_SHARE`` of the year (half, by
default) is not in the basis case; the fit can then only add a heater or
cooler at its end. On the crude preheat train with the residue rundown
running only in the peak period (0.2 of the year) the design cost
332,284 $/y against 308,263 on basis "all", and nothing said so.
"""

from __future__ import annotations

import json

import pytest

from OpenPinch import PinchProblem
from OpenPinch.resources import read_sample_case


def test_the_median_basis_warns_of_a_stream_it_leaves_out():
    source = json.loads(read_sample_case("crude_preheat_train_multiperiod.json"))
    stream = next(
        s for s in source["streams"] if s["name"] == "Atmospheric residue rundown"
    )
    stream["heat_flow"]["values"] = [0, 0, 1150]
    problem = PinchProblem(source, project_name="crude")

    with pytest.warns(UserWarning, match="Atmospheric residue rundown") as caught:
        problem.design._median_basis(None)

    (warning,) = [w for w in caught if "residue rundown" in str(w.message)]
    assert "20 %" in str(warning.message)
    assert "design_basis='all'" in str(warning.message)


def test_a_stream_left_out_of_the_designed_cluster_is_named_with_its_year():
    # Two clusters: turndown and base (0.8 of the year), and peak (0.2).
    # Only the dominant cluster is designed; a stream that runs only in
    # peak is in no part of it, though it runs 20 % of the year.
    source = json.loads(read_sample_case("crude_preheat_train_multiperiod.json"))
    stream = next(
        s for s in source["streams"] if s["name"] == "Atmospheric residue rundown"
    )
    stream["heat_flow"]["values"] = [0, 0, 1150]
    problem = PinchProblem(source, project_name="crude")

    with pytest.warns(UserWarning, match="Atmospheric residue rundown") as caught:
        problem.design._median_basis(2)

    (warning,) = [w for w in caught if "residue rundown" in str(w.message)]
    assert "Crude Unit" in str(warning.message)
    assert "20 % of the year" in str(warning.message)
