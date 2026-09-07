# Integration test instructions

Execute notebook 08 in a fresh kernel and temporary working directory using the
development interpreter. Verify distinct finite HP/RF GCC and NLP traces,
positive residual loads, successful bounded placement and unchanged serialized
residual_basis before/after placement. Keep executed outputs in temporary
verification artifacts; do not overwrite the source notebook.

The real workflow tests exercise target ownership, default/explicit graph
selection, segmented utility scaling, utility allocation and candidate placement.
Run the solver-marked suite separately with pytest -m solver. The existing
installed-wheel smoke verifies public APIs, resources and TESPy target/map
compatibility when that optional dependency is installed.
