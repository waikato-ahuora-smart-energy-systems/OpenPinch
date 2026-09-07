# Residual utility design

Validate a retained scalar HPR handle before conversion. Snapshot numerical residual, physical boundary, provenance identity and digest; create streams=[] with copied utility definitions and one period. Existing direct allocation dispatch uses that snapshot before process preprocessing. Utility placement builds its snapshot and candidate allocations directly from this same frozen basis. Reconstruction and serialization retain it. HPR is never rerun.

## Testable Properties

Round trip: canonical JSON retains the complete frozen basis. Invariants: all candidate profiles equal the frozen profile, independent source/result state, selected period and units. Oracle: detached utility allocation agrees with HPR residual accounting. Idempotence: repeated conversions have equal basis. Stateful: sequences of allocation, serialization and utility changes keep the basis and source unchanged. Commutativity/induction N/A.
