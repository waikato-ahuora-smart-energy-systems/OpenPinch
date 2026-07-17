"""Internal segment record owned and transacted by :class:`Stream`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import numpy as np

from ..enums import FluidPhase
from ..stream import Stream
from ..value import Value


class StreamSegment(Stream):
    """One ordered linear interval owned by a physical parent :class:`Stream`."""

    def __init__(self, *args, segment_index: int | None = None, **kwargs):
        self._owner: Stream | None = None
        self._segment_index = segment_index
        super().__init__(*args, **kwargs)

    @property
    def parent(self) -> Stream | None:
        """Return the owning parent stream, if the segment is attached."""
        return self._owner

    @property
    def segment_index(self) -> int | None:
        """Return the stable zero-based position within the parent profile."""
        return self._segment_index

    @property
    def segment_id(self) -> str:
        """Return a stable parent-qualified segment identifier."""
        if self._owner is None or self._segment_index is None:
            return self.name
        return f"{self._owner.name}.S{self._segment_index + 1}"

    @property
    def dt_cont_multiplier(self) -> float:
        """Return the zone shift multiplier inherited from the parent."""
        return self._dt_cont_multiplier

    @dt_cont_multiplier.setter
    def dt_cont_multiplier(self, value: float) -> None:
        if self._owner is not None:
            raise ValueError(
                "Segment zone shift multiplier is controlled by its parent stream."
            )
        Stream.dt_cont_multiplier.fset(self, value)

    @property
    def active(self) -> bool:
        """Return the activity inherited from the parent when attached."""
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        if self._owner is not None:
            raise ValueError("Segment activity is controlled by its parent stream.")
        Stream.active.fset(self, value)

    @property
    def is_process_stream(self) -> bool:
        """Return the process/utility role inherited from the parent."""
        return self._is_process_stream

    @is_process_stream.setter
    def is_process_stream(self, value: bool) -> None:
        if self._owner is not None:
            raise ValueError("Segment role is controlled by its parent stream.")
        Stream.is_process_stream.fset(self, value)

    @property
    def fluid_name(self) -> Optional[str]:
        """Return fluid metadata inherited from the parent."""
        return self._fluid_name

    @fluid_name.setter
    def fluid_name(self, value: Optional[str]) -> None:
        if self._owner is not None:
            raise ValueError(
                "Segment fluid metadata is controlled by its parent stream."
            )
        Stream.fluid_name.fset(self, value)

    @property
    def fluid_phase(self) -> Optional[str]:
        """Return phase metadata inherited from the parent."""
        return self._fluid_phase

    @fluid_phase.setter
    def fluid_phase(self, value: Optional[str | FluidPhase]) -> None:
        if self._owner is not None:
            raise ValueError(
                "Segment phase metadata is controlled by its parent stream."
            )
        Stream.fluid_phase.fset(self, value)

    def set_value_attr(
        self,
        attr_name: str,
        value: float | Value | np.ndarray | Mapping | None,
        update_derived: bool = True,
    ) -> None:
        if self._owner is not None and not self._owner._syncing_segments:
            self._owner.update_segment(self._segment_index, **{attr_name: value})
            return
        super().set_value_attr(attr_name, value, update_derived=update_derived)

    def set_value_attr_at_idx(
        self,
        attr_name: str,
        value: float | Value | np.ndarray = None,
        idx: int = 0,
        update_derived: bool = True,
    ):
        if self._owner is not None and not self._owner._syncing_segments:
            candidate = self._detached_value_for_period(attr_name, value, idx)
            self._owner.update_segment(self._segment_index, **{attr_name: candidate})
            return
        return super().set_value_attr_at_idx(
            attr_name,
            value,
            idx=idx,
            update_derived=update_derived,
        )

    def set_period_context(
        self,
        period_ids: dict[str, int] | list[str] | tuple[str, ...] | None,
        weights: np.ndarray | list[float] | tuple[float, ...] | None,
        num_periods: int | None,
    ) -> None:
        if self._owner is not None:
            expected_ids = self._owner.period_ids
            expected_weights = self._owner.weights
            same_ids = period_ids == expected_ids
            same_weights = (weights is None and expected_weights is None) or (
                weights is not None
                and expected_weights is not None
                and np.array_equal(np.asarray(weights), expected_weights)
            )
            if (
                not same_ids
                or not same_weights
                or num_periods != self._owner.num_periods
            ):
                raise ValueError(
                    "Segment period context is controlled by its parent stream."
                )
        super().set_period_context(period_ids, weights, num_periods)

    def _detached_value_for_period(self, attr_name: str, value, idx: int) -> Value:
        internal_name = self._resolve_attr_name(attr_name)
        current = getattr(self, internal_name)
        if current is None:
            current = Value(0.0, unit=self._VALUE_UNITS[internal_name])
        candidate = Value(current)
        target_size = self._period_vector_size()
        if candidate.num_periods == 1 and target_size > 1:
            candidate = Value(
                np.full(target_size, float(candidate.value), dtype=float),
                unit=candidate.unit,
            )
        candidate[idx if candidate.num_periods > 1 else 0] = value
        return candidate
