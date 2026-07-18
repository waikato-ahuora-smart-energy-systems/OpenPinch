# Unit 2 NFR Requirements

- **Learnability**: public method names must reveal the engineering outcome and
  appear through IDE completion without importing selection enums.
- **Predictability**: the same explicit arguments and prepared problem state must
  resolve to the same service graph and ordered result.
- **Transparency**: validation errors must name conflicting public arguments and
  explain the valid call shape.
- **Performance**: `all_heat_integration()` must retain one dependency-aware
  zone traversal; all-period replay must support deterministic serial execution
  and bounded parallel execution.
- **State safety**: observation must not mutate or solve; replay must restore the
  caller's selected-period state even when execution fails.
- **Import discipline**: root imports remain limited to `PinchProblem` and
  `PinchWorkspace`; optional solver/plot backends remain lazy.
- **Testability**: argument precedence and period ordering must have generated
  invariant coverage with normal shrinking and fixed CI reproduction.
