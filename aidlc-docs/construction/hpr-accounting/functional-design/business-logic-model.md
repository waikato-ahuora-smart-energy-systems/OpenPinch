# Accounting logic

Select fraction times available net heating (HP) or cooling (RF), bounded by
available duty. Preserve the selected background region near the pinch.
HP useful heating is a ceiling; unused cold incurs economic cost only. RF
selected cooling is a service obligation under existing feasibility terms.
Cycle condenser duty equals evaporator duty plus net work. HP condenser base
must not grow when an ambient sink is introduced.

Form the residual from the background with ambient minus the HPR cascade.
Rebase this combined cascade to minimum zero before pocket removal: a floating
nonzero minimum must not make pinch detection fail and inflate both loads.
Store the resulting profiles on their exact grid with ProblemTable.update,
which aligns grids; do not discard updates when new breakpoints appear.
Keep full precision throughout accounting. Graph creation follows residual
postprocessing so summaries and profiles share one grid.

## Testable Properties

Invariants: fraction bounds, finite equal-length residual columns, cycle balance.
Oracle: additive cascade offset cannot change utility demand after rebasing.
Idempotence: pocket-free normalized residual remains unchanged when processed
again. Round trip: immutable numerical records retain tuples and units.
Commutativity/induction N/A: no order-independent operations or recursion added.
Stateful properties N/A here; conversion lifecycle is owned by unit 3.
