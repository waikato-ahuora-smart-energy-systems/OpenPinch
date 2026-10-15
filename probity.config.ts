import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  defineConfig,
  enforceTdd,
  forbidCommandPattern,
  forbidContentPattern,
} from '@nizos/probity'
import type { Rule } from '@nizos/probity/rules'

const ROOT = dirname(fileURLToPath(import.meta.url))

// `git commit`, also with global options such as `git -C dir commit`.
const GIT_COMMIT = /\bgit(?:\s+-[Cc]\s+\S+|\s+--?[\w-]+(?:=\S+)?)*\s+commit\b/
// `git push` with --force, --force-with-lease, -f (also combined, e.g. -uf)
// or a +refspec.
const GIT_FORCE_PUSH =
  /\bgit(?:\s+-[Cc]\s+\S+|\s+--?[\w-]+(?:=\S+)?)*\s+push\b[^;&|]*(?:\s--force|\s-[a-zA-Z]*f\b|\s\+\S)/
// pytest run as a command (at the start or after && ; | or a subshell,
// optionally through `uv run [--flags]` and/or `[path/]python -m`, e.g.
// `.venv/bin/python -m pytest`), not just mentioned in a commit message or
// echo; --collect-only / --co does not count.
const PYTEST_RUN =
  /(?:^|[;&|(])\s*(?:uv\s+run(?:\s+--?[\w-]+(?:=\S+)?)*\s+)?(?:\S*\/)?(?:python3?(?:\.\d+)?\s+-m\s+)?pytest\b(?![^;&|]*\s--(?:collect-only|co)\b)/m
// pytest's final summary line, e.g. "1 failed, 2 passed in 0.40s". Judging
// only this line keeps other tools' output (ruff's "Found 1 error.") out.
const PYTEST_SUMMARY = /^.*\b\d+ (?:passed|failed|errors?)\b.* in [\d.]+s\b.*$/gm
const PYTEST_FAILED = /\b\d+ (?:failed|errors?)\b/

// The part of a command that is actually run: its first line (a heredoc body
// such as a commit message follows on later lines) without quoted strings
// (an unclosed quote runs to the end of the line).
function commandText(command: string): string {
  const firstLine = command.split('\n', 1)[0] ?? ''
  return firstLine.replace(/'[^']*(?:'|$)|"[^"]*(?:"|$)/g, '')
}

function isPytestRun(command: string): boolean {
  return PYTEST_RUN.test(commandText(command))
}

function pytestPassed(output: string): boolean {
  const summary = output.match(PYTEST_SUMMARY)?.at(-1)
  return (
    summary !== undefined &&
    /\b\d+ passed\b/.test(summary) &&
    !PYTEST_FAILED.test(summary)
  )
}

// Commits need a pytest run after the latest write whose output reports
// passing tests and no failures or errors.
function requirePassingPytest(): Rule {
  return async function requirePassingPytest(action, ctx) {
    if (action.kind !== 'command' || !GIT_COMMIT.test(action.command)) {
      return { kind: 'pass' }
    }
    if (!existsSync(join(ROOT, 'tests'))) return { kind: 'pass' }
    const history = (await ctx?.history?.()) ?? []
    for (let i = history.length - 1; i >= 0; i--) {
      const event = history[i]
      if (event?.kind === 'write') break
      if (event?.kind === 'command' && isPytestRun(event.command)) {
        if (pytestPassed(event.output)) return { kind: 'pass' }
        break
      }
    }
    return {
      kind: 'violation',
      reason:
        'Run the tests that cover your change (`pytest`) after the latest write and get them passing before committing.',
    }
  }
}

const noSuppressionComments = forbidContentPattern({
  match: /#\s*(?:noqa|ruff:\s*noqa|type:\s*ignore|pyright:\s*ignore|mypy:\s*ignore)/,
  reason: 'Fix the lint or type issue rather than suppressing it.',
})

export default defineConfig({
  rules: [
    requirePassingPytest(),
    forbidCommandPattern({
      match: GIT_FORCE_PUSH,
      reason: 'Do not force-push. Make a new commit instead.',
    }),
    {
      files: ['OpenPinch/**', 'tests/**'],
      rules: [
        enforceTdd({
          maxEvents: 12,
          maxContentChars: 10000,
          instructions: (defaults) => `${defaults}

### Project rules (OpenPinch)

- Tests use pytest and live under tests/, mirroring the package layout
  (tests/analysis/... for OpenPinch/analysis/...), in files named test_*.py.
- Property-based tests use Hypothesis; CI runs with --hypothesis-seed=20260715.
- Optional or slow tests carry markers declared in pytest.ini (solver, tespy,
  performance, docs, synthesis, tutorial_profile); the unit lane is
  -m "not solver and not tespy and not performance and not docs".
- Numerical assertions use pytest.approx or numpy.testing with an explicit
  rel/abs tolerance, never bare float equality. Changing a published or
  benchmark number needs a test showing why the new value is right.
- A bug fix starts with a regression test that reproduces the bug.
- Updating an existing test's expected value to match corrected behaviour
  (with the reason in a comment) counts as the failing test for that fix.`,
        }),
        forbidContentPattern({
          match: /\b(?:pytest\.(?:mark\.)?|mark\.)(?:skip|xfail)\b/,
          reason:
            'Do not skip or xfail tests. Fix the test, or use skipif with a concrete environment condition.',
        }),
      ],
    },
    {
      files: ['OpenPinch/**', 'tests/**', 'scripts/**'],
      rules: [noSuppressionComments],
    },
    {
      // pyproject.toml already carries a small, reviewed lint policy, so it is
      // not covered here; changes to it are reviewed in the pull request.
      files: ['ruff.toml', '.ruff.toml', 'setup.cfg'],
      rules: [
        forbidContentPattern({
          match: /^\s*(?:extend-)?(?:per-file-)?ignores?\s*=|per-file-ignores\]/m,
          reason:
            'Do not add lint ignores in configuration. Fix the issue, or ask the user to change the lint policy.',
        }),
      ],
    },
    {
      files: ['**/*.md'],
      rules: [
        forbidContentPattern({
          match: /\p{Extended_Pictographic}/u,
          reason: 'No emojis in documentation.',
        }),
      ],
    },
  ],
})
