# Utility Placement Uncapped Notebook Code Generation Plan

This corrective TDD plan removes per-utility duty upper bounds from the single
executable utility-placement notebook without changing the public API.

## Scope

- Notebook 19 shall run uncapped Process and Site utility-placement workflows.
- The small `search_options` limits remain to keep the tutorial executable.
- The optional `maximum_duties` API and its specialist tests remain unchanged.
- RTD shall document upper bounds separately rather than applying them in the
  primary runnable notebook workflow.
- The notebook remains assertion-free and uses standard GCC/TSP plots.

## Execution steps

- [x] Step 1 - Change the notebook packaging contract in RED to reject duty
  upper-bound setup and arguments in notebook 19.
- [x] Step 2 - Remove bounds from the generator and align the RTD runnable
  example and notebook interpretation.
- [x] Step 3 - Regenerate and execute notebook 19; verify uncapped Process and
  Site duties, exact retarget replay, and standard plots.
- [x] Step 4 - Run focused and complete quality gates, review the diff, update
  completion records, and commit the amendment to `develop`.

## Extension compliance

PBT behavior is unchanged because this is a presentation-input amendment. The
existing example and property tests for maximum-duty behavior remain in place.
Security and Resiliency remain disabled and N/A.
