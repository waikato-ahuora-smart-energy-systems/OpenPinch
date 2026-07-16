# Segment Batch Update and Pricing Code Generation Summary

## Result

The approved segmented-stream mutation and utility-pricing change is complete.
One transactional primitive now backs sparse batch, single-segment, and uniform
parent-to-child mutations. Segment prices remain independent, while a segmented
utility parent's displayed price is the duty-weighted value for each period and
its cost is the exact sum of child costs.

## Application Changes

- Added public sparse `Stream.update_segments(...)` with atomic validation,
  rollback, order preservation, and one revision/cache commit.
- Preserved child prices when the segmented parent constructor omits `price`;
  explicit parent price assignment remains a deliberate broadcast.
- Added nested segmented/profile utility preparation with child-before-parent
  price precedence and legacy flat-utility compatibility.
- Added ordered utility price and cumulative-cost tensors to HEN preparation.
- Added exact piecewise utility cost mappings for integer-capable and active-
  segment NLP solver paths, extraction, verification, and reporting.
- Kept HEN utility selection parent-based and limited to the current selected hot
  and cold utility; multi-utility selection remains deferred.

## Tests and Documentation

- Added deterministic and Hypothesis coverage for transaction rollback,
  multiperiod price conservation, nested utility validation, stable identities,
  and exact partial/boundary/full-profile HEN costs.
- Updated the domain model, input-format guide, HEN synthesis guide, and
  capability matrix.
- No duplicate application files or alternate model implementations were
  created.

## Extension Compliance

- Security Baseline: disabled; N/A.
- Resiliency Baseline: disabled; N/A.
- Property-Based Testing (Partial): compliant through reusable constrained
  generators, standard shrinking, and reproducible seed `20260715`.

## Content Validation

This Markdown contains no Mermaid, ASCII diagram, JSON, or YAML block. Headings,
lists, paths, inline code, and punctuation were checked for CommonMark-compatible
rendering.
