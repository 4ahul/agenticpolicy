<div align="center">

# AgenticPolicy

**Safety guardrails for AI agents.**

Add authorization, approval gates, rate limits and data-exfiltration protection to an existing agent — without rewriting it.

[![test](https://github.com/4ahul/agenticpolicy-/actions/workflows/test.yml/badge.svg)](https://github.com/4ahul/agenticpolicy-/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/agenticpolicy)](https://pypi.org/project/agenticpolicy/)
[![python](https://img.shields.io/pypi/pyversions/agenticpolicy)](pyproject.toml)
[![dependencies](https://img.shields.io/badge/core%20dependencies-0-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-black)](LICENSE)

</div>

```python
from agenticpolicy import Policy, GuardedAgent

policy = Policy(agent_id="support_bot")
policy.allow("read", ["crm:ticket", "kb:*"])
policy.allow("write", ["crm:ticket"], conditions={"field": "status"})
policy.deny("delete", ["*"])
policy.prevent_exfiltration(max_output_kb=50)

guarded = GuardedAgent.create(
    model=llm, tools=tools, policy=policy, context={"user_id": "u_42"}
)
result = guarded.invoke({"messages": [("user", "Close ticket 8891")]})
```

---

## The problem

Agents get handed the same credentials their author has. A support bot that should read tickets can usually also delete customers, because nothing between the model and the tool checks whether *this* call was one anybody intended.

The failure mode isn't a jailbreak. It's an agent doing exactly what a confusing prompt asked it to, with production access.

> Prompt instructions are not a control. A tool the model can reach is a tool the model can be talked into using.

## What it does

`agenticpolicy` sits between the agent and its tools. Every call is checked before it runs; every result is checked before it goes back to the model.

```
                  ┌──────────────────────────────────────┐
   agent ────────►│  evaluate: context → deny → allow    │
   wants to       │  → conditions → arg scan → gates     │
   call a tool    │  → budget                            │
                  └───────────────┬──────────────────────┘
                                  │ allowed
                                  ▼
                            ┌──────────┐
                            │   tool   │
                            └────┬─────┘
                                 │ returns
                  ┌──────────────▼───────────────────────┐
   model sees ◄───│  check_output: size cap, PII scan,   │
   the result     │  redaction                            │
                  └──────────────┬───────────────────────┘
                                 │
                                 ▼
                          SQLite audit log
                    (every decision, allowed or not)
```

| Control | What it stops |
|---|---|
| **Default-deny authorization** | Any action on any resource nobody explicitly allowed |
| **Deny rules** | Protected resources, whatever else the policy says |
| **Conditions** | Writes to the wrong field, actions in the wrong environment |
| **Approval gates** | High-impact calls proceeding without a human |
| **Budgets** | A retry loop deploying forty times |
| **Exfiltration scanning** | PII and secrets leaving through tool args *or* tool output |
| **Size caps** | `SELECT *` walking out of the warehouse |
| **Audit log** | Not knowing what your agent did |

## Install

```bash
pip install agenticpolicy                    # core, zero dependencies
pip install "agenticpolicy[langchain]"       # + LangChain adapter
pip install "agenticpolicy[integrations]"    # + all three adapters
```

Python 3.10+. The core is standard library only, and `import agenticpolicy` does not pull in LangChain — there's a test that asserts it.

## Five-minute start

**1. Write a policy**

```python
from agenticpolicy import Policy

policy = Policy(agent_id="support_bot")
policy.require_context(["user_id", "ticket_id"])       # traceability
policy.allow("read", ["crm:ticket", "docs:*", "kb:*"])
policy.allow("write", ["crm:ticket"], conditions={"field": "status"})
policy.deny("write", ["crm:customer", "billing:*"])
policy.deny("delete", ["*"])
policy.prevent_exfiltration(max_output_kb=50)
```

**2. Check it says what you think**

```python
print(policy.explain())
assert policy.validate() == []   # catches unreachable gates and shadowed rules
```

**3. Wrap your agent**

```python
from agenticpolicy import GuardedAgent

guarded = GuardedAgent.create(              # LangChain 1.x
    model=llm,
    tools=[lookup_ticket, close_ticket],
    policy=policy,
    context={"user_id": "u_42", "ticket_id": "T-8891"},
    resource_map={"lookup_ticket": "crm:ticket"},   # explicit beats inferred
)

result = guarded.invoke({"messages": [("user", "What's the status of my ticket?")]})
print(guarded.report())     # what was blocked, and why
```

Blocked calls return an explanatory string the model reads as an observation, so the agent tries a permitted path instead of crashing. Pass `on_violation="raise"` to handle `PolicyViolation` yourself.

```
1 tool call(s) evaluated, 1 blocked
  - delete_record (delete crm:customer): Denied by rule deny_1
```

## Frameworks

<table>
<tr><th align="left">LangChain</th></tr>
<tr><td>

```python
# 1.x — guard the tools, then build the agent
from agenticpolicy import GuardedAgent
guarded = GuardedAgent.create(model=llm, tools=tools, policy=policy)

# 0.x AgentExecutor, or anything exposing .tools
guarded = GuardedAgent(executor, policy=policy)
```
</td></tr>
<tr><th align="left">LlamaIndex</th></tr>
<tr><td>

```python
from agenticpolicy import GuardedLlamaIndexAgent
guarded = GuardedLlamaIndexAgent.from_tools(tools, llm=llm, policy=policy)
guarded.chat("How many tickets are open?")
```
</td></tr>
<tr><th align="left">LangGraph</th></tr>
<tr><td>

```python
from agenticpolicy.integrations.langgraph_ import GuardedToolNode
graph.add_node("tools", GuardedToolNode(tools, policy=policy))
```
</td></tr>
<tr><th align="left">Anything else</th></tr>
<tr><td>

```python
from agenticpolicy.integrations.base import ToolGuard
guard = ToolGuard(policy, context={"user_id": "u_42"})
safe_delete = guard.wrap(delete_record, resource="crm:customer", action="delete")
```
</td></tr>
</table>

All four share one code path, so behaviour is identical across frameworks. Each adapter is ~80 lines.

## Resource patterns

Resources are `provider:type`. Two wildcards:

| Pattern | Matches | Does not match |
|---|---|---|
| `salesforce:lead` | `salesforce:lead` | `salesforce:contact` |
| `crm:*` | `crm:ticket` | `crm:ticket:comment`, `salesforce:lead` |
| `crm:**` | `crm:ticket`, `crm:ticket:comment` | `salesforce:lead` |
| `*:lead` | `salesforce:lead`, `hubspot:lead` | `salesforce:contact` |
| `*` | everything | — |

`*` stays inside one segment; `**` crosses `:`. Matching is case-insensitive, and regex metacharacters are literal — `crm.ticket` does **not** match `crm:ticket`. A guardrail library that silently widens a rule is worse than none.

## Approval gates and budgets

```python
policy.allow("execute", ["ci:deploy"])                       # required first
policy.require_approval("execute", ["ci:deploy"], budget={"per_day": 5})
```

> **The mistake everyone makes.** Evaluation is default-deny, so an approval gate on an action no allow rule permits **never fires** — the call is denied before the gate is reached. The policy reads like it gates deploys; it forbids them. `policy.validate()` catches exactly this.

For a cap with no human in the loop:

```python
policy.rate_limit("execute", ["ci:*"], per_hour=10)
```

Budgets are rolling windows (`per_minute`, `per_hour`, `per_day`) or per-task (`per_task`, keyed on `context["task_id"]`). Charged only when a call actually executes, so evaluating a call you then abandon doesn't spend the quota.

## Exfiltration protection

```python
policy.prevent_exfiltration(
    max_output_kb=50,
    block_patterns=[r"\b\d{3}-\d{2}-\d{4}\b"],   # defaults cover SSN, cards,
    scan_args=True,                              # emails, AWS keys, private
    scan_output=True,                            # keys, secret assignments
    redact=False,                                # True masks instead of blocking
)
```

Both directions are scanned, and **outbound arguments matter more than they look**: an agent talked into exfiltrating data puts the payload in a tool call, not in a response.

Findings name the pattern that matched and how many times — never the matched value, so the audit log doesn't become its own leak.

## Audit log

```python
from agenticpolicy import EventStore, PolicyEngine

store = EventStore("audit.db")
engine = PolicyEngine(policy, store=store)
guarded = GuardedAgent.create(model=llm, tools=tools, engine=engine)
```

```bash
agenticpolicy audit --db audit.db --blocked-only
agenticpolicy summary --db audit.db --hours 24
```

```
TIME                 AGENT         ACTION   RESOURCE          VERDICT   REASON
2026-08-30 11:04:12  support_bot   delete   crm:ticket        DENY      Denied by rule deny_3
2026-08-30 11:04:09  support_bot   write    billing:refund    DENY      Denied by rule deny_2
```

Append-only by convention. `sqlite3`, not SQLAlchemy — the part you most want present in production carries no third-party dependency and cannot fail to import.

## Policies in CI

Save a policy to JSON and assert its behaviour, so a well-meaning widening shows up in review:

```bash
agenticpolicy test policy.json write github:main   --expect deny
agenticpolicy test policy.json execute ci:deploy   --expect needs_approval
```

Non-zero exit on mismatch. This and `validate()` are the parts no comparable library has.

## Pre-built policies

```python
from agenticpolicy import PrebuiltRules

PrebuiltRules.least_privilege_support_bot()
PrebuiltRules.code_agent_with_gates(deploys_per_day=5)
PrebuiltRules.data_analyst(max_output_kb=50)
PrebuiltRules.read_only()
PrebuiltRules.human_in_the_loop()
PrebuiltRules.rate_limited_agent(calls_per_hour=100)
PrebuiltRules.no_data_exfiltration()
```

Starting points to extend, not compliance guarantees. `agenticpolicy catalog` lists them.

## Examples

Three runnable scripts, no API keys needed:

```bash
python examples/support_bot.py     # least privilege + audit trail
python examples/code_agent.py      # approval gates + deploy budget
python examples/data_analyst.py    # bulk-export protection
```

## Where this fits

This is **authorization**, not alignment, and not output filtering. It constrains what an agent *can* do — the tractable half of the problem. It will not tell you whether the agent should have wanted to.

Three honest limits:

- **In-process enforcement is a strong control against a confused agent and a weak one against an attacker with code execution.** Anything that can run arbitrary Python in your process can bypass an in-process check. See [docs/architecture.md](docs/architecture.md).
- **Resource inference is a convenience.** `infer_resource("salesforce_read_lead")` guesses `salesforce:lead`. Fine in a prototype; in production pass `resource_map` explicitly so renaming a tool can't silently change which rules apply. Unrecognised names fall back to `execute`, the most restricted action.
- **Pattern scanning catches patterned data.** SSNs and API keys have shapes. A customer list does not — that's what `max_output_kb` is for.

## Docs

| | |
|---|---|
| [Getting started](docs/getting_started.md) | Install to first blocked call |
| [Policy reference](docs/policy_reference.md) | Every rule type, condition operator and pattern |
| [Architecture](docs/architecture.md) | Evaluation order, threat model, what this does not defend against |
| [Contributing](CONTRIBUTING.md) | Dev setup, test layout |

## License

MIT
