# Delivery infrastructure design plan

NFR Design was approved with `Continue`.

- [x] Inspect existing release triggers, publishing permissions, and environments.
- [x] Map validation and explicit release entry points to workflow owners.
- [x] Define runners, artifact storage, permissions, concurrency, and diagnostics.
- [x] Define deployment order and required-check/publisher migration safeguards.
- [x] Verify requirement coverage and stage compliance.
- [x] Obtain approval before Code Generation planning (`Continue`).

## Applicability and clarification assessment

Deployment remains GitHub Actions plus TestPyPI/PyPI; no new provider choice.
Compute remains hosted runners with the existing solver-specific runner.
Storage is CI artifacts and release assets; no database. Workflow dependencies
provide sequencing; no messaging service. HTTPS outbound access to existing
services suffices; no gateway, load balancer, or network topology change.
Monitoring uses job summaries and retained reports. Shared infrastructure is
the existing repository, its environments, rulesets, and trusted publishers;
no cross-unit service is introduced, so no separate shared-infrastructure
artifact is necessary.

There are no further policy questions. Publisher/environment configuration
must be verified at rollout rather than assumed from YAML. That verification
does not authorize changing remote settings or publishing a package.
