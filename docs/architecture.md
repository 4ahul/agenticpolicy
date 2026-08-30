# Architecture

How `agenticpolicy` is put together, why it's shaped this way, and what it does not cover.

---

## Layers

```
┌───────────────────────────────────────────────────────────────┐
│  Adapters      langchain_.py   llamaindex_.py   langgraph_.py │
│                        all delegate to ToolGuard              │
├───────────────────────────────────────────────────────────────┤
│  ToolGuard     integrations/base.py                           │
│                infer → evaluate → run → scan output → commit  │
├───────────────────────────────────────────────────────────────┤
│  PolicyEngine  core/engine.py       BudgetTracker             │
│                evaluate() / check_output() / commit()         │
├───────────────────────────────────────────────────────────────┤
│  Policy        core/policy.py       core/matching.py          │
│                rules, conditions, validation                  │
├───────────────────────────────────────────────────────────────┤
│  Types         core/types.py        (stdlib dataclasses only) │
├───────────────────────────────────────────────────────────────┤
│  Audit         audit/store.py       (stdlib sqlite3 only)     │
└───────────────────────────────────────────────────────────────┘
```

The dependency direction is strictly downward. Nothing in `core/` imports an adapter, and nothing in `core/` imports a third-party package.

### Why the framework-agnostic middle layer

All three adapters do the same five things in the same order. Putting that sequence in `ToolGuard` rather than in each adapter means:

- The interesting logic is testable without LangChain, LlamaIndex or LangGraph installed. The suite runs in a fraction of a second on a bare interpreter.
- A fourth framework — or a bare function, or an MCP server — is a thin file, not a reimplementation.
- A bug fixed in the sequence is fixed everywhere at once.

---

## The two-phase evaluation

Checks split by when they're possible:

```
                    ┌──────────────────────┐
   agent decides    │  evaluate()          │
   to call a tool ──▶  1 required context  │──▶ denied  ──▶ "[BLOCKED] ..."
                    │  2 deny rules        │                (tool never runs)
                    │  3 allow rules       │
                    │  4 outbound arg scan │
                    │  5 approval gates    │
                    │  6 budgets           │
                    └──────────┬───────────┘
                               │ allowed
                               ▼
                        ┌─────────────┐
                        │  tool runs  │
                        └──────┬──────┘
                               ▼
                    ┌──────────────────────┐
                    │  check_output()      │
                    │  7 size limit        │──▶ denied ──▶ "[BLOCKED] ..."
                    │  8 pattern scan      │              (data never reaches
                    └──────────┬───────────┘               the model)
                               │ allowed
                               ▼
                        ┌─────────────┐
                        │  commit()   │  charge budgets
                        └─────────────┘
```

The original design folded all of this into one `evaluate()` call. That can't work: output doesn't exist before the tool runs, so a one-phase design can only ever check arguments — which is how a library ends up shipping a `_check_exfiltration_risk` that returns `0.0` and a policy that quietly enforces nothing.

Splitting it also makes the guarantee honest. A pre-call denial means the side effect never happened. A post-call denial means the tool ran but the data didn't reach the model. Those are different promises, and conflating them misleads.

### Why `commit()` is separate

Budgets are charged after a call executes, not when it's evaluated. Evaluating a call that's then abandoned — the agent changed its mind, an upstream error fired, a framework retried — must not spend the user's quota. Charging at evaluation time makes a rate limit slowly drift down under retry loops, in a way nobody notices until the budget mysteriously runs out early.

---

## Data model

```python
ToolCall        # what the agent wants to do
  agent_id, tool_name, resource, action, args, context, timestamp

PolicyDecision  # what the engine decided
  allowed, effect, reason, rule_applied, requires_approval,
  risk_score, findings, metadata

PolicyEvent     # what gets written to the audit log
  PolicyEvent.from_decision(call, decision)
```

`Effect` has three values — `ALLOW`, `DENY`, `NEEDS_APPROVAL` — rather than a bare boolean. "Blocked because it's forbidden" and "blocked pending a human" call for different handling downstream: one is an error, the other is a queue.

`findings` names the pattern that matched and the match count, never the matched value. An audit log that quotes the SSN it caught is its own data leak.

---

## Resource matching

Patterns are `provider:type` with two wildcards: `*` within a segment, `**` across `:`.

The implementation escapes regex metacharacters before substituting wildcards, so `crm.ticket` matches only the literal resource `crm.ticket`. The obvious implementation — `pattern.replace("*", ".*")` and match — has two failure modes that both widen rules silently:

- `.` in a pattern becomes "any character", so `crm.ticket` matches `crm:ticket`.
- `*` spans `:`, so `crm:*` matches `crm:ticket:comment` when you meant one level.

A guardrail library that widens rules by accident is worse than none, because it's trusted. Compiled patterns are cached (`lru_cache`), so correctness costs nothing on the hot path — 1000 evaluations run in well under a second, against a 5ms-per-call target.

---

## Conditions

Evaluated against `context` first, then `args`. Context winning matters: `args` come from the model, `context` comes from your application. If arguments could override context, an agent could satisfy `{"role": "admin"}` by passing `role="admin"` — the condition would check the model's claim about itself.

Conditions fail closed. A missing field or a type mismatch (`"abc"` against `{"gte": 1000}`) fails the condition rather than raising, because an exception thrown from inside a policy check crashes the agent — turning a guardrail into an outage.

---

## Budgets

`BudgetTracker` keeps timestamps per `(agent, rule, window)` in a deque, trimmed on read. A budget of "10 per hour" means ten calls in any rolling hour, not ten since the process started.

The clock is injectable, so budget-window behaviour is tested by advancing a fake clock rather than sleeping through an hour. Tests that need real time don't get written, and behaviour that isn't tested doesn't stay correct.

`per_task` has no time window; it counts against `context["task_id"]` and resets on a new task.

---

## Audit store

`sqlite3` from the standard library, not SQLAlchemy. The audit trail is the part you most want present in production, and it shouldn't be able to fail to import because a dependency conflicts with something else in the environment.

Append-only by convention: nothing updates or deletes a row except `purge()`, which exists for retention policies and says so.

Store failures are swallowed — deliberately, and only here. If the disk fills, the agent keeps working and stops logging. The alternative is a full disk taking down production because the audit writer raised.

Threading: one connection per thread via `threading.local`, except `:memory:`, which shares one connection because an in-memory database lives only as long as its connection.

---

## Optional dependencies

`import agenticpolicy` imports no framework. The top-level names `GuardedAgent`, `GuardedLlamaIndexAgent` and `GuardedToolNode` resolve through a module-level `__getattr__`, so they're importable from the package root but only load their framework when actually used.

A missing framework raises `IntegrationNotInstalled` with the exact pip command, rather than an `ImportError` from three frames down.

There's a test asserting `langchain` is absent from `sys.modules` after importing the package, because this is the kind of property that regresses the moment someone adds a convenience import at the top of a file.

---

## Guarding by replacement, not patching

The original design patched `tool.invoke` in place inside a `try/finally`. Three problems:

- It mutates objects the caller still holds.
- If the process dies mid-run, the patch leaks into whatever else holds that tool.
- Concurrent runs double-patch, and the restore order decides what's left.

The adapters instead build new tool objects wrapping the originals — same name, description and argument schema, so the model sees no difference and prompts need no rewriting. The caller's tools are untouched.

---

## Performance

| Operation | Target | Actual |
|---|---|---|
| Policy evaluation | <5ms | ~0.05ms (1000 evals well under 1s) |
| Audit write | — | one indexed INSERT |
| Import (core) | — | no third-party imports |

Pattern compilation is cached; evaluation is a linear walk over rules, which is the right complexity for policies of tens of rules. A policy with thousands would want a resource-prefix index — not built, because nothing needs it yet.

---

## What this does not do

Stated plainly, because a security library that oversells its coverage is dangerous.

**It's authorization, not alignment.** It constrains what an agent *can* do. It has nothing to say about whether the agent should have wanted to.

**Resource inference is a heuristic.** `infer_resource("salesforce_read_lead")` guesses from the name. Fine for a prototype; in production pass `resource_map` so a rename can't change which rules apply. Unrecognised names fall back to `execute`, the most restricted action, so unmapped tools fail closed.

**Pattern scanning catches patterned data.** SSNs, cards and API keys have shapes. A customer list doesn't — which is what `max_output_kb` is for, and why the size cap is the real control in the data-analyst policy rather than the regexes.

**Enforcement is in-process.** The guard runs in your Python process, alongside the agent. It's a strong control against a confused agent and a weak one against an attacker with code execution in that process. Server-side enforcement — evaluating policy in a service the agent can't modify — is the natural next step and is not built.

**Output checks happen after execution.** By the time output is rejected, the tool has run. What's prevented is the data reaching the model. For side effects you must prevent, use deny rules or approval gates, which run before.

**Budgets are per-process.** `BudgetTracker` lives in memory. Several replicas of the same agent each get their own budget. Shared limits need a shared backend — Redis, or the database — and that isn't built.

---

## Extension points

| You want to | Do this |
|---|---|
| Support another framework | Write an adapter over `ToolGuard`; ~80 lines |
| Store audit elsewhere | Pass any object with `log_event(PolicyEvent)` as `store` |
| Add a condition operator | Extend `_compare` in `core/matching.py` |
| Add a pre-built policy | Add a `@staticmethod` to `PrebuiltRules`; the catalog picks it up |
| Test budget behaviour | Inject a fake clock: `PolicyEngine(policy, clock=fake)` |
| Distribute budgets | Replace `BudgetTracker` with a Redis-backed implementation |
