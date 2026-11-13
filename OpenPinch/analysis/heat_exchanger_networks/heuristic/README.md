# HEN Heuristic Pinch Design

## Overview
A rules-based (Linnhoff) pinch design method. It proposes network topologies for each pinch side; the pinch design side model in `../models` realises each proposal with only its matches allowed. This package is pure Python decision logic: it reads a set-up, unsolved `PinchDecompModel` and builds no model of its own.

## Interface
- `pinch_side`, `PinchSide`, `SideStream`, `PINCH_TOLERANCE` (`side_streams.py`): one side's decision data. Each active stream carries per-period loads, the CP at its pinch end, whether it reaches the pinch, and its temperature against the heat counted from its pinch end; the side carries pair approaches, the recovery pairs the side model can place, and the utilities each stream can use.

## Invariants
- Only streams active on the side in some period are listed. A stream absent in a period has zero load, zero CP and no pinch contact there [side_streams.py:_side_stream].
- The pinch end is a hot stream's outlet above the pinch and its inlet below; a cold stream's inlet above and its outlet below. A stream reaches the pinch in a period when its pinch-end temperature is within `PINCH_TOLERANCE` K of the pinch temperature shifted by its own contribution [side_streams.py:_side_stream].
- A segmented stream's pinch CP is that of the clipped segment at its pinch end, and its temperature follows the clipped piecewise profile [side_streams.py:_profile_points].
- `load` and `pinch_cp` are period-weighted means; `pinch_cp` averages only over periods where the stream is active [side_streams.py:SideStream].

## Side Effects
- None: reads model attributes and solver arrays only.

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks/test_heuristic_side_streams.py`
