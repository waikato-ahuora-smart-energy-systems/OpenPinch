Direct vs Indirect Integration
==============================

One of the most important distinctions in OpenPinch is the difference between
direct integration and indirect integration.

Direct Integration
------------------

Direct integration is the classical process-level pinch problem:

- process hot streams are matched against process cold streams
- heat recovery happens directly inside the targeted system boundary
- utilities cover the remaining deficits and surpluses

In OpenPinch, this is usually the first solve and the baseline answer. It is
what most users mean by a direct pinch target.

Indirect Integration
--------------------

Indirect integration operates one level up:

- solved subzones are reduced to net thermal behavior
- those net thermal segments are aggregated at a larger system boundary
- utility-to-utility balancing is used to recover heat across zones through the
  wider utility system rather than only by direct process-to-process exchange

This is the foundation for Total Process and Total Site workflows.

Why The Answers Differ
----------------------

Direct and indirect results differ because they do not use the same exchange
mechanism or system boundary.

Direct integration answers:
   How much recovery can be achieved inside this zone through direct thermal
   matching?

Indirect integration answers:
   How much additional benefit emerges when solved subzones exchange energy
   through a utility-mediated higher-level system?

Hierarchy Matters
-----------------

OpenPinch models these workflows through nested zones:

- unit operation
- process zone
- site
- extended multiscale labels where needed

Direct integration is usually meaningful at unit-operation or process-zone
scope. Indirect integration becomes meaningful when multiple solved subzones
can be aggregated into a higher-level system.

Practical Reading Rule
----------------------

When comparing direct and indirect answers:

- compare like with like by target row name
- compare utility targets first
- use graphs to explain why the change occurred
- do not assume an indirect answer is always better; it is only better if the
  larger system boundary creates additional feasible recovery

OpenPinch Surfaces
------------------

Common user-facing routes are:

- `problem.target()` for the default high-level workflow
- `problem.target.direct_heat_integration(...)`
- `problem.target.indirect_heat_integration(...)`
- the CLI `run` and `graph` workflows

Recommended Follow-On Pages
---------------------------

- :doc:`zones-streams-utilities-and-targets`
- :doc:`graphs-and-interpretation`
- :doc:`../guides/zonal-and-total-site-workflows`
