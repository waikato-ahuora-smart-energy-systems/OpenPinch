"""How exported plot files load plotly.js."""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem


@pytest.fixture(scope="module")
def solved() -> PinchProblem:
    problem = PinchProblem("basic_pinch.json")
    problem.target.direct_heat_integration()
    return problem


def test_gallery_can_share_one_plotly_js_file(solved, tmp_path):
    solved.plot.export_gallery(
        tmp_path, plot=solved.plot.grand_composite_curve, include_plotlyjs="directory"
    )

    plots = [path for path in tmp_path.glob("*.html") if path.name != "index.html"]
    assert plots
    assert [path.name for path in tmp_path.glob("*.js")] == ["plotly.min.js"]
    for path in plots:
        assert 'src="plotly.min.js"' in path.read_text(encoding="utf-8")


def test_export_can_load_plotly_js_from_its_cdn(solved, tmp_path):
    [path] = solved.plot.export(
        tmp_path, plot=solved.plot.grand_composite_curve, include_plotlyjs="cdn"
    )

    html = path.read_text(encoding="utf-8")
    assert "cdn.plot.ly" in html
    assert len(html) < 100_000


@pytest.mark.parametrize("include_plotlyjs", [False, "require", "plotly.js"])
def test_export_rejects_other_ways_to_load_plotly_js(
    solved, tmp_path, include_plotlyjs
):
    with pytest.raises(ValueError, match="include_plotlyjs"):
        solved.plot.export(tmp_path, include_plotlyjs=include_plotlyjs)
    with pytest.raises(ValueError, match="include_plotlyjs"):
        solved.plot.export_gallery(tmp_path, include_plotlyjs=include_plotlyjs)


def test_catalog_names_the_plot_method_of_each_graph(solved):
    catalog = solved.plot.catalog()

    gcc = catalog[catalog["Graph Type"] == "Grand Composite Curve"]
    assert set(gcc["Plot Method"]) == {"grand_composite_curve"}
    for method in catalog["Plot Method"].dropna():
        assert callable(getattr(solved.plot, method))
