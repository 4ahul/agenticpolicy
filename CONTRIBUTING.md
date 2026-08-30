# Contributing

Thanks for looking. This is a security library, so a few things matter more here than in a typical project.

## Setup

```bash
git clone https://github.com/rahulsagar/agenticpolicy
cd agenticpolicy
pip install -e ".[dev,integrations]"
pytest
```

The suite runs in well under a second with no network and no API keys. If it doesn't, something is wrong.

## Before you open a PR

```bash
pytest              # all green
ruff check src tests
black src tests examples
mypy src
```

## What a good change looks like

**Every behaviour change needs a test.** In a guardrail library the bugs that matter are the ones that quietly *widen* a rule — a pattern that matches more than intended, a check that returns `True` on an error path. Those don't surface in manual testing, because nothing looks broken.

**Prefer failing closed.** When a check can't reach a definite answer — a missing field, a malformed pattern, a type mismatch — deny rather than allow. Every existing check does this; the tests assert it.

**Never let a guard raise into the agent.** A policy check that throws turns a guardrail into an outage. Catch, deny, and record it as a finding. `scan_text` handles a malformed user regex this way, and there's a test for it.

**Findings must not quote what they caught.** An audit log that echoes the SSN it blocked is its own data leak. Name the pattern and the count.

**Don't add a required dependency.** The core imports only the standard library, and `import agenticpolicy` must not pull in a framework — there's a test asserting that. Framework code goes in `integrations/` behind an optional extra.

## Where things live

| Path | What |
|---|---|
| `core/types.py` | Dataclasses. No third-party imports, ever. |
| `core/matching.py` | Resource patterns and conditions. |
| `core/policy.py` | The DSL, validation, serialization. |
| `core/engine.py` | Evaluation and budgets. |
| `core/rules.py` | Pre-built policies. |
| `audit/store.py` | SQLite audit log. `sqlite3` only. |
| `integrations/base.py` | `ToolGuard` — the shared guarding sequence. |
| `integrations/*_.py` | Thin framework adapters over `ToolGuard`. |

New framework support belongs in `integrations/`, delegating to `ToolGuard`. If you find yourself reimplementing evaluate-run-scan-commit, put the shared part in `base.py` instead.

## Adding a pre-built policy

Add a `@staticmethod` to `PrebuiltRules` with a one-line summary as the first docstring line — `catalog()` and the CLI read it. There's a parametrized test asserting every pre-built has a docstring and produces no `validate()` warnings, so a policy with an unreachable gate fails CI.

## Reporting a vulnerability

Please don't open a public issue. Email the maintainer with details and a reproduction, and give it a reasonable window before disclosing.

Findings especially worth reporting: a resource pattern matching more than it should, a condition evaluating to allow when it should deny, a way to reach a tool without evaluation, or an audit record containing data it captured.

## Code style

Black, 100 columns. Type hints on public functions. Docstrings that say *why*, not just what — the reason a check is ordered where it is, or fails closed the way it does, is the part a reader can't recover from the code.
