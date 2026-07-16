"""Private Plotly adapters used by the network grid renderer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _SplitGroup:
    start_x: float
    end_x: float
    left_connector_x: float
    right_connector_x: float


class _PlotlyLine:
    def __init__(self, xdata, ydata, **kwargs):
        self.xdata = tuple(xdata)
        self.ydata = tuple(ydata)
        self.kwargs = kwargs
        self.openpinch_tooltip: str | None = None
        self._plotly_trace_indices: list[int] = []

    def get_xdata(self):
        return self.xdata

    def get_ydata(self):
        return self.ydata


class _PlotlyText:
    def __init__(
        self,
        text: str,
        *,
        xy: tuple[float, float],
        position: tuple[float, float],
        ha: str,
        va: str,
    ):
        self.xy = xy
        self._position = position
        self._text = text
        self._ha = ha
        self._va = va

    def get_text(self) -> str:
        return self._text

    def get_position(self) -> tuple[float, float]:
        return self._position

    def get_ha(self) -> str:
        return self._ha

    def get_va(self) -> str:
        return self._va


class _PlotlyAxes:
    def __init__(self, fig: Any, graph_objects: Any):
        self.fig = fig
        self.go = graph_objects
        self.lines: list[_PlotlyLine] = []
        self.texts: list[_PlotlyText] = []
        self.stream_bounds: list[tuple[float, float]] = []
        self._yticks: list[float] = []

    def arrow(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
        *,
        lw: float,
        color: str,
        **_kwargs,
    ) -> None:
        x_end = x + dx
        y_end = y + dy
        self.stream_bounds.append((min(x, x_end), max(x, x_end)))
        self.fig.add_trace(
            self.go.Scatter(
                x=[x, x_end],
                y=[y, y_end],
                mode="lines",
                line={"color": color, "width": lw},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        symbol = "triangle-right" if dx >= 0 else "triangle-left"
        self.fig.add_trace(
            self.go.Scatter(
                x=[x_end],
                y=[y_end],
                mode="markers",
                marker={"symbol": symbol, "size": max(16, lw * 3.2), "color": color},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    def add_line(self, line: _PlotlyLine) -> None:
        self.lines.append(line)
        mode = "lines+markers" if line.kwargs.get("marker") else "lines"
        marker = None
        if line.kwargs.get("marker"):
            marker = {
                "size": _plotly_marker_size(line.kwargs.get("markersize", 18)),
                "color": line.kwargs.get("markerfacecolor", line.kwargs.get("color")),
            }
        trace = self.go.Scatter(
            x=list(line.xdata),
            y=list(line.ydata),
            mode=mode,
            line={
                "color": line.kwargs.get("color", "black"),
                "width": line.kwargs.get("lw", 2),
                "dash": "dash" if line.kwargs.get("linestyle") == "dashed" else "solid",
            },
            marker=marker,
            hoverinfo="skip",
            showlegend=False,
        )
        self.fig.add_trace(trace)
        line._plotly_trace_indices.append(len(self.fig.data) - 1)

    def update_line_hover(self, line: _PlotlyLine, tooltip: str) -> None:
        for index in line._plotly_trace_indices:
            self.fig.data[index].update(
                hovertemplate=tooltip.replace("\n", "<br>") + "<extra></extra>",
                hoverinfo=None,
            )

    def text(self, x: float, y: float, text: str, **kwargs) -> _PlotlyText:
        return self.annotate(text, xy=(x, y), xytext=(0, 0), **kwargs)

    def annotate(
        self,
        text: str,
        *,
        xy: tuple[float, float],
        xytext: tuple[float, float] = (0, 0),
        **kwargs,
    ) -> _PlotlyText:
        ha = kwargs.get("ha", kwargs.get("horizontalalignment", "left"))
        va = kwargs.get("va", kwargs.get("verticalalignment", "baseline"))
        label = _PlotlyText(text, xy=xy, position=xytext, ha=ha, va=va)
        self.texts.append(label)
        font = {"color": kwargs.get("color", "black")}
        if kwargs.get("fontsize") is not None:
            font["size"] = kwargs["fontsize"]
        annotation = {
            "x": xy[0],
            "y": xy[1],
            "text": text.replace("$^\\circ$", "°"),
            "showarrow": False,
            "xshift": xytext[0],
            "yshift": xytext[1],
            "xanchor": _plotly_xanchor(ha),
            "yanchor": _plotly_yanchor(va),
            "font": font,
        }
        if kwargs.get("bgcolor") is not None:
            annotation["bgcolor"] = kwargs["bgcolor"]
        if kwargs.get("borderpad") is not None:
            annotation["borderpad"] = kwargs["borderpad"]
        self.fig.add_annotation(**annotation)
        return label

    def set_yticks(self, ticks, labels) -> None:
        self._yticks = list(ticks)
        self.fig.update_yaxes(tickmode="array", tickvals=self._yticks, ticktext=labels)

    def get_yticks(self):
        return self._yticks

    def set_xticks(self, _ticks) -> None:
        self.fig.update_xaxes(showticklabels=False, ticks="")

    def set_xlim(self, left: float, right: float) -> None:
        self.fig.update_xaxes(range=[left, right])


def _plotly_marker_size(markersize: float) -> float:
    return max(12.0, min(float(markersize) * 0.42, 30.0))


def _plotly_xanchor(ha: str) -> str:
    return {"left": "left", "right": "right", "center": "center"}.get(ha, "left")


def _plotly_yanchor(va: str) -> str:
    return {
        "top": "top",
        "bottom": "bottom",
        "center": "middle",
        "center_baseline": "middle",
        "baseline": "middle",
    }.get(va, "middle")
