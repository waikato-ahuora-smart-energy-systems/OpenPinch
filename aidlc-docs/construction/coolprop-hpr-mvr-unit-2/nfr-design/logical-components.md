# Unit 2 Logical Components

| Component | Responsibility | Lifetime |
|---|---|---|
| Public budget resolver | apply precedence and strict validation | invocation |
| CoolProp capability preflight | produce detached per-stage capability facts | invocation |
| Warm-start evaluator | evaluate normalized starts before backend | invocation |
| Search evaluation cache | memoize exact search points | invocation |
| Candidate merger | deterministically deduplicate/rank points | invocation |
| Failure accumulator | count failures and retain at most 16 representatives | invocation |
| HPR optimisation adapter | translate budget to generic optimiser options | invocation |
| Multiperiod coordinator | apply the same budget/preflight to every period | invocation |

No component persists data or owns an external process.
