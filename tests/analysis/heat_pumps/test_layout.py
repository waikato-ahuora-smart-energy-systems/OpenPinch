import numpy as np
import pytest

from OpenPinch.analysis.heat_pumps.common.layout import HPRoptVectorLayout

from .helpers import _base_args


def test_hpr_opt_vector_layout_packs_sections_in_canonical_order():
    layout = HPRoptVectorLayout(
        n_cond=2,
        n_evap=2,
        n_subcool=1,
        n_heat_split=2,
        n_cool_split=1,
        n_ihx=2,
        n_misc=1,
    )

    x = layout.pack(
        x_cond=[1.0, 2.0],
        x_evap=[3.0, 4.0],
        x_subcool=[5.0],
        x_heat_split=[8.0, 9.0],
        x_cool_split=[10.0],
        x_ihx=[11.0, 12.0],
        x_misc=[13.0],
    )
    parts = layout.unpack(x)

    np.testing.assert_allclose(
        x,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]),
    )
    np.testing.assert_allclose(parts["x_cond"], np.array([1.0, 2.0]))
    np.testing.assert_allclose(parts["x_evap"], np.array([3.0, 4.0]))
    np.testing.assert_allclose(parts["x_subcool"], np.array([5.0]))
    np.testing.assert_allclose(parts["x_heat_split"], np.array([8.0, 9.0]))
    np.testing.assert_allclose(parts["x_cool_split"], np.array([10.0]))
    np.testing.assert_allclose(parts["x_ihx"], np.array([11.0, 12.0]))
    np.testing.assert_allclose(parts["x_misc"], np.array([13.0]))


def test_non_brayton_backends_share_canonical_opt_vector_prefix():
    args = _base_args(n_cond=2, n_evap=2)
    layouts = [
        HPRoptVectorLayout(n_cond=int(args.n_cond), n_evap=int(args.n_evap)),
        HPRoptVectorLayout(n_cond=int(args.n_cond), n_evap=int(args.n_evap)),
        HPRoptVectorLayout(
            n_cond=int(args.n_cond),
            n_evap=int(args.n_evap),
            n_subcool=int(args.n_cond),
            n_heat_split=int(args.n_cond),
            n_ihx=int(args.n_cond),
        ),
        HPRoptVectorLayout(
            n_cond=int(args.n_cond),
            n_evap=int(args.n_evap),
            n_subcool=int(args.n_cond),
            n_heat_split=int(args.n_cond),
            n_cool_split=max(int(args.n_evap) - 1, 0),
            n_ihx=int(args.n_cond) + int(args.n_evap) - 1,
        ),
    ]

    for layout in layouts:
        assert layout.cond_slice == slice(0, 2)
        assert layout.evap_slice == slice(2, 4)

    assert layouts[0].size == 4
    assert layouts[1].size == 4
    assert layouts[2].subcool_slice == slice(4, 6)
    assert layouts[2].heat_split_slice == slice(6, 8)
    assert layouts[2].ihx_slice == slice(8, 10)
    assert layouts[3].subcool_slice == slice(4, 6)
    assert layouts[3].heat_split_slice == slice(6, 8)
    assert layouts[3].cool_split_slice == slice(8, 9)
    assert layouts[3].ihx_slice == slice(9, 12)


def test_hpr_opt_vector_layout_validates_counts_and_empty_layout():
    with pytest.raises(ValueError, match="n_cond"):
        HPRoptVectorLayout(n_cond=-1)

    empty = HPRoptVectorLayout(
        n_cond=0,
        n_evap=0,
        n_subcool=0,
        n_heat_split=0,
        n_cool_split=0,
        n_ihx=0,
        n_misc=0,
    )

    assert empty.size == 0
    assert empty.pack().size == 0
    assert empty.build_bounds() == []


def test_hpr_opt_vector_layout_rejects_bad_vector_block_and_bounds_sizes():
    layout = HPRoptVectorLayout(n_cond=2, n_evap=1)

    with pytest.raises(ValueError, match="Expected optimisation vector"):
        layout.unpack(np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="x_cond"):
        layout.pack(x_cond=[0.1])
    with pytest.raises(ValueError, match="x_cond bounds"):
        layout.build_bounds(x_cond=[(0.0, 1.0)])
    assert layout.build_bounds(x_cond=[(0.0, 0.5), (0.5, 1.0)])[0:2] == [
        (0.0, 0.5),
        (0.5, 1.0),
    ]
