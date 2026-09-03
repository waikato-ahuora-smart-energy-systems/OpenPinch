# Integration Test Instructions

## Static Integration

Parse every GitHub Actions workflow and run the release documentation contract:

```bash
ruby -e 'require "yaml"; Dir[".github/workflows/*.yml"].each { |path| YAML.safe_load(File.read(path), [], [], true) }'
uv run --no-sync pytest tests/packaging/test_docs_consistency.py -q
```

## Live Pull-Request Acceptance

For the next same-repository PR targeting `main` whose candidate equals the
base, confirm this sequence in GitHub Actions:

1. `bump-version` advances the configured version and pushes one commit.
2. `release-version` starts only after bump evaluation and validates the latest
   head successfully.
3. A rerun with the forward version creates no additional bump commit.
4. `pr-gate` succeeds after all applicable validation jobs pass.

No external cleanup is required.
