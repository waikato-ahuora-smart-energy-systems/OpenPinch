import numpy as np

from OpenPinch.services.heat_pump_integration.common.layout import HPRoptVectorLayout

from ..helpers import _base_args


def test_hpr_opt_vector_layout_packs_sections_in_canonical_order():
    layout = HPRoptVectorLayout(
        n_cond=2,
        n_evap=2,
        n_subcool=1,
        n_heat=1,
        n_cool=1,
        n_ihx=2,
        n_misc=1,
    )

    x = layout.pack(
        x_amb=0.1,
        x_cond=[1.0, 2.0],
        x_evap=[3.0, 4.0],
        x_subcool=[5.0],
        x_heat=[6.0],
        x_cool=[7.0],
        x_ihx=[8.0, 9.0],
        x_misc=[10.0],
    )
    parts = layout.unpack(x)

    np.testing.assert_allclose(
        x,
        np.array([0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
    )
    assert parts["x_amb"] == 0.1
    np.testing.assert_allclose(parts["x_cond"], np.array([1.0, 2.0]))
    np.testing.assert_allclose(parts["x_evap"], np.array([3.0, 4.0]))
    np.testing.assert_allclose(parts["x_subcool"], np.array([5.0]))
    np.testing.assert_allclose(parts["x_heat"], np.array([6.0]))
    np.testing.assert_allclose(parts["x_cool"], np.array([7.0]))
    np.testing.assert_allclose(parts["x_ihx"], np.array([8.0, 9.0]))
    np.testing.assert_allclose(parts["x_misc"], np.array([10.0]))


def test_non_brayton_backends_share_canonical_opt_vector_prefix():
    args = _base_args(n_cond=2, n_evap=2)
    layouts = [
        HPRoptVectorLayout(n_cond=int(args.n_cond), n_evap=int(args.n_evap)),
        HPRoptVectorLayout(n_cond=int(args.n_cond), n_evap=int(args.n_evap)),
        HPRoptVectorLayout(
            n_cond=int(args.n_cond),
            n_evap=int(args.n_evap),
            n_subcool=int(args.n_cond),
            n_heat=int(args.n_cond),
            n_ihx=int(args.n_cond),
        ),
        HPRoptVectorLayout(
            n_cond=int(args.n_cond),
            n_evap=int(args.n_evap),
            n_subcool=int(args.n_cond),
            n_heat=int(args.n_cond),
            n_cool=max(int(args.n_evap) - 1, 0),
            n_ihx=int(args.n_cond) + int(args.n_evap) - 1,
        ),
    ]

    for layout in layouts:
        assert layout.amb_slice == slice(0, 1)
        assert layout.cond_slice == slice(1, 3)
        assert layout.evap_slice == slice(3, 5)

    assert layouts[0].size == 5
    assert layouts[1].size == 5
    assert layouts[2].subcool_slice == slice(5, 7)
    assert layouts[2].heat_slice == slice(7, 9)
    assert layouts[2].ihx_slice == slice(9, 11)
    assert layouts[3].subcool_slice == slice(5, 7)
    assert layouts[3].heat_slice == slice(7, 9)
    assert layouts[3].cool_slice == slice(9, 10)
    assert layouts[3].ihx_slice == slice(10, 13)
