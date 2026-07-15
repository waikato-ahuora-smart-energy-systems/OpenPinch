# Segmented Variable-CP Requirement Decisions

All material questions were answered in the approved implementation plan.

1. Which data is authoritative for a nested profile?
   - [Answer]: Profile values are authoritative. Infer parent aggregates and validate duplicated parent values.
2. How are multiperiod segment identities stabilized?
   - [Answer]: Use the union of normalized cumulative-duty breakpoints across periods.
3. How is exchanger area subdivided?
   - [Answer]: Use ordered duty-aligned slices split at every hot or cold segment boundary.
4. How is segment continuity enforced?
   - [Answer]: Preserve input order and require each segment target temperature to equal the next segment supply temperature in every period after unit normalization.
5. Are existing flat stream rows grouped automatically?
   - [Answer]: No. Existing flat rows remain independent physical streams.
6. Which extensions are enabled?
   - [Answer]: Security Baseline: No. Resiliency Baseline: No. Property-Based Testing: Partial.
