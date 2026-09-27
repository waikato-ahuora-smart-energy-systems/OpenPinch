#!/usr/bin/env bash
# Run the same checks as .github/workflows/ci.yml on this machine.
#
#   scripts/ci_local.sh              # default: lint, unit (with coverage), docs, build + smoke
#   scripts/ci_local.sh quick        # lint + unit tests, no coverage, stop at first failure
#   scripts/ci_local.sh all          # every lane, including tespy, notebooks, performance, solver
#   scripts/ci_local.sh unit solver  # any lanes by name
#
# Lanes: lint unit docs tespy performance solver notebooks build
# Needs uv. The solver lane downloads the IDAES solver binaries on first use.
set -euo pipefail
cd "$(dirname "$0")/.."

SEED=20260715
UNIT_MARKERS="not solver and not tespy and not performance and not docs"
DEFAULT_LANES=(lint unit docs build)
ALL_LANES=(lint unit docs tespy performance solver notebooks build)
mkdir -p reports

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
pytest_lane() { # name, marker expression, extra pytest args...
  local name=$1 markers=$2
  shift 2
  uv run --no-sync pytest --hypothesis-seed="$SEED" -m "$markers" \
    --junitxml="reports/$name.xml" --durations=20 "$@"
}

lane_lint() {
  uv run --no-sync ruff check .
  uv run --no-sync python scripts/check_lockfile_version.py
  if command -v actionlint > /dev/null; then
    actionlint
  else
    echo "actionlint not installed; skipping workflow lint (brew install actionlint)"
  fi
}
lane_quick() {
  lane_lint
  pytest_lane unit "$UNIT_MARKERS" -x -q
}
lane_unit() {
  uv run --no-sync coverage run --branch --source=OpenPinch -m pytest \
    --hypothesis-seed="$SEED" -m "$UNIT_MARKERS" --junitxml=reports/unit.xml --durations=20
  uv run --no-sync coverage report --fail-under=95
}
lane_docs() { pytest_lane docs "docs"; }
lane_tespy() { pytest_lane tespy "tespy and not tutorial_profile"; }
# Base and interactive notebooks already run in the unit lane.
lane_notebooks() { pytest_lane notebooks "tutorial_profile and (tespy or solver)"; }
lane_performance() { pytest_lane performance "performance"; }
lane_solver() {
  uv run --no-sync python -c "
from pyomo.environ import SolverFactory
import idaes  # noqa: F401  (puts the IDAES binaries on PATH)
missing = [s for s in ('couenne', 'ipopt') if not SolverFactory(s).available(exception_flag=False)]
raise SystemExit(1 if missing else 0)" || uv run --no-sync idaes get-extensions
  pytest_lane solver "solver and not tutorial_profile"
}
lane_build() {
  # Build once, then install the wheel (not the checkout) into a separate
  # environment and smoke-test the core surface, as the CI smoke job does.
  rm -rf dist
  SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)" uv build --out-dir dist
  UV_PROJECT_ENVIRONMENT=.venv-smoke uv sync --frozen --no-default-groups --no-install-project
  uv pip install --python .venv-smoke --no-deps --reinstall dist/*.whl
  .venv-smoke/bin/python scripts/optional_install_smoke.py core
  .venv-smoke/bin/python scripts/artifact_install_smoke.py --surface core
}

lanes=("$@")
if [ ${#lanes[@]} -eq 0 ]; then
  lanes=("${DEFAULT_LANES[@]}")
elif [ "${lanes[0]}" = all ]; then
  lanes=("${ALL_LANES[@]}")
fi

say "Syncing the locked dev environment"
uv sync --frozen --group dev --extra tespy

failed=()
for lane in "${lanes[@]}"; do
  if ! declare -F "lane_$lane" > /dev/null; then
    echo "Unknown lane: $lane (choose from quick ${ALL_LANES[*]} all)" >&2
    exit 2
  fi
  say "$lane"
  start=$SECONDS
  # A subshell with errexit, so the first failing command fails the lane.
  set +e
  (set -e; "lane_$lane")
  status=$?
  set -e
  if [ "$status" -eq 0 ]; then
    echo "$lane passed in $((SECONDS - start))s"
  else
    echo "$lane FAILED after $((SECONDS - start))s"
    failed+=("$lane")
  fi
done

say "Summary"
if [ ${#failed[@]} -eq 0 ]; then
  echo "All lanes passed: ${lanes[*]}"
else
  echo "Failed: ${failed[*]}"
  exit 1
fi
