"""loops.py exports the constant-utility helper it now holds.

``with_constant_utilities`` moved from the retired ``heat_pumps.py`` into
``loops.py``, its only caller, and the retrofit README documents it there;
the module's ``__all__`` lists it so a wildcard import finds it too.
"""

from __future__ import annotations


def test_loops_exports_with_constant_utilities():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import loops

    assert "with_constant_utilities" in loops.__all__
