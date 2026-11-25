"""A model with its interior cuts installed goes between processes.

With ``HENS_MAX_PARALLEL`` above 1 a worker process sends its solved
models back pickled. The interior approach check was stored on the model
as a lambda, which does not pickle, so a design of a case with a
segmented stream raised ``PicklingError`` as soon as it used workers.
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace

from OpenPinch.analysis.heat_exchanger_networks.models._base import (
    interior_approach,
)


def test_a_model_with_interior_cuts_installed_pickles():
    owner = SimpleNamespace(
        integers=True,
        N_periods=0,
        I=1,
        J=1,
        _segmented_hot_parents=[True],
        _segmented_cold_parents=[False],
    )
    interior_approach.install_interior_cuts(owner, [[[1000.0]]])

    restored = pickle.loads(pickle.dumps(owner))

    # The restored model checks itself: no period, so nothing to cut.
    assert restored._add_lazy_cuts() is False
