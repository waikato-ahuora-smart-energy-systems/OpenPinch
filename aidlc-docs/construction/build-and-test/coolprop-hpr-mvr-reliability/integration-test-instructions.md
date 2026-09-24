# Integration Test Instructions

## Service Boundary Gate

Run heat-pump analysis together with public application and notebook contracts:

```bash
uv run pytest tests/analysis/heat_pumps tests/contracts \
  tests/application/test_hpr_derived_case_api.py \
  tests/application/test_hpr_performance_map_accessor.py \
  tests/application/test_hpr_period_batch_boundaries.py \
  tests/application/test_analysis_reliability.py \
  tests/packaging/test_notebooks.py -q
```

This covers CoolProp VC/cascade/parallel/VC+MVR objectives, multiperiod
aggregation, detached public outputs, process-MVR components, and generated
tutorial contracts.

## Public Workflow Gate

The audit adds an unmocked end-to-end regression gate:

```bash
uv run pytest tests/application/test_coolprop_hpr_audit.py \
  tests/analysis/heat_pumps/test_hpr_audit_regressions.py -q
```

The public matrix covers direct and utility heat-pump/refrigeration paths,
VC+MVR, non-scalar cascade/parallel/MVR records, detached public snapshots and
invalid-fluid atomicity. The generated cache model and penalty oracle are
fixed-seed properties, not mocked thermodynamic feasibility proofs.

Execute every code cell of packaged notebooks 09 and 11 in clean namespaces.
Notebook 09 must produce a successful scalar CoolProp target and performance
map. Notebook 11 must produce direct process-MVR stage results and equal serial
and parallel multiperiod results. Typed physical HPR failures may be reported;
unexpected runtime defects must abort.

## Artifact Gate

Install the built wheel outside the checkout and run:

```bash
python scripts/artifact_install_smoke.py --surface tespy
```

The import must resolve from the temporary installation and packaged HPR
contracts, simulator assets, notebooks, sample cases, CLI, and public workflow
must all succeed.
