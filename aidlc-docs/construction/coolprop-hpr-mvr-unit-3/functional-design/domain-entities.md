# Unit 3 Domain Entities

## DirectGasMVRFallbackDiagnostic

Frozen detached value with `code`, bounded summary, source stream, period index,
stage index, and fluid. Code is exactly `dry_stage` or `reduced_profile`.

## DirectGasMVRStageError

`ValueError`-compatible domain exception with stable reason code plus detached
source stream, period, stage, and fluid attributes. Raw engine details remain
only on `__cause__`.

## DirectGasMVRStageResult extension

Additive tuple of fallback diagnostics. Empty on ordinary success; never
contains an exception or engine object.
