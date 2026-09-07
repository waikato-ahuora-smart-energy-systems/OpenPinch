# HPR accounting implementation

R1/R3/R5; HPR-US-1, 2, 4. Fraction validation now accepts finite values in [0, 1].
HP condenser sizing cannot expand its selected heating ceiling by adding an
ambient sink. Unused low-grade heat already had no HP feasibility penalty;
regressions preserve that distinction from refrigeration. Carnot results now
retain their actual feasibility penalty.

Utility-system residuals now include ambient exchange in the same background
used for physical accounting; a real refrigeration regression exposed a
14.011 kW omission that otherwise prevented balanced-composite placement.
Residual cascades are rebased before pocket removal, and newly introduced
breakpoints are retained in the problem table. Finite immutable records carry
available, selected and achieved service, cycle duties, work, ambient exchanges,
exact residual coordinates and the physical thermal boundary.

Owners: domain/hpr.py, configuration_fields.py, targets.py; analysis/heat_pumps
service.py, common/postprocessing.py, common/shared.py and Carnot targeting.
Tests: test_hpr_residual_regressions.py, existing cascade/parallel/targeting tests,
and the real application workflow. The focused numerical/graph/contract gate
passed 125 tests; the later Carnot gate passed 31 tests. Final aggregate evidence
is in construction/build-and-test/hpr-notebook-observations/.

Final affected verification after the indirect ambient correction: 731 tests
passed in 145.60 seconds. See the integrated Build and Test summary for final
aggregate coverage and the preserved notebook exclusions.

## Extension compliance

PBT-01 through PBT-10: compliant across the integrated change. Designs identify
properties; generated residual cases exercise the independent offset oracle,
nonnegative/finite profiles and idempotence. Graph and frozen-input JSON round
trips preserve data. Generated operation sequences compare allocation, roundtrip
and price edits with the unchanged reference basis after every operation,
including empty sequences. Constrained Hypothesis generators retain shrinking;
seed 20260715 matches CI. Real solver, graph and segmented-utility examples
complement properties. Commutativity/induction are N/A for these operations.
Security and Resiliency are disabled in aidlc-state.md and were skipped.
