# Business Logic Model

Input normalization constructs candidate segments, standardizes their units, validates period counts and ordered continuity, and only then commits them to a parent stream. Aggregate state is recalculated from the validated profile. Expanded numeric projection emits child rows for segmented parents and ordinary rows for unsegmented parents, retaining parent metadata in both cases.
