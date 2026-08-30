# Getting Started

Fifteen minutes from install to a guarded agent with an audit trail.

---

## Install

```bash
pip install agenticpolicy                  # core, no dependencies
pip install "agenticpolicy[langchain]"     # + LangChain adapter
```

Python 3.10 or newer.

---

## 1. Your first policy

A policy is a list of rules. Nothing is permitted unless a rule permits it.

```python
from agenticpolicy import Policy

policy = Policy(agent_id="my_first_agent")
policy.allow("read", ["docs:*"])
policy.deny("delete", ["*"])
```

Read it back before trusting it:

```python
print(policy.explain())
```

```
Policy for agent 'my_first_agent'
  [DENY] delete on *
  [ALLOW] read on docs:*
```

---

## 2. Guard some tools

You don't need an agent framework to see this working. `ToolGuard` wraps plain functions.

```python
from agenticpolicy.integrations.base import ToolGuard

def read_docs_page(path: str) -> str:
    return f"contents of {path}"

def delete_docs_page(path: str) -> str:
    return f"deleted {path}"

guard = ToolGuard(policy)
safe_read = guard.wrap(read_docs_page)
safe_delete = guard.wrap(delete_docs_page)

print(safe_read(path="intro.md"))
# contents of intro.md

print(safe_delete(path="intro.md"))
# [BLOCKED] Denied by rule deny_0
```

The delete never ran. That's the whole idea: the guard sits in front of the tool, not in the prompt.

---

## 3. Understand what just happened

`read_docs_page` was mapped to the resource `docs:page` and the action `read`, inferred from the function name. Inference is a convenience — see step 6 for why you should be explicit in production.

Ask the guard what it did:

```python
print(guard.report())
```

```
2 tool call(s) evaluated, 1 blocked
  - delete_docs_page (delete docs:page): Denied by rule deny_0
```

---

## 4. Add real controls

Four more lines take this from a demo to something you'd deploy.

```python
policy = Policy(agent_id="support_bot")

# Every call must say who and what it's for. Untraceable calls get nothing.
policy.require_context(["user_id", "ticket_id"])

policy.allow("read", ["crm:ticket", "kb:*"])

# Writes only to the status field — not the owner, not the customer record.
policy.allow("write", ["crm:ticket"], conditions={"field": "status"})

policy.deny("write", ["crm:customer", "billing:*"], description="no PII writes")
policy.deny("delete", ["*"], description="support never deletes")

# Cap how much data can leave, and scan it for PII on the way.
policy.prevent_exfiltration(max_output_kb=50)
```

Supply the required context once:

```python
guard = ToolGuard(policy, context={"user_id": "u_42", "ticket_id": "T-8891"})
```

---

## 5. Check the policy says what you meant

```python
warnings = policy.validate()
assert warnings == [], warnings
```

`validate()` catches the mistake almost everyone makes at least once:

```python
policy = Policy("x")
policy.allow("read", ["github:*"])
policy.require_approval("execute", ["ci:deploy"])    # looks like a deploy gate

print(policy.validate())
# ['approve_1: approval gate on execute ['ci:deploy'] is unreachable — no allow
#   rule permits it, so those calls are denied outright. Add
#   policy.allow("execute", ['ci:deploy']) if you meant to gate them.']
```

Evaluation is default-deny, so the call is refused at the allow stage and the gate is never reached. The policy reads like it gates deploys; it actually forbids them. Put `assert policy.validate() == []` in your tests.

---

## 6. Be explicit about resources

Name inference is for prototyping. In production, map tools to resources yourself:

```python
guard = ToolGuard(
    policy,
    context={"user_id": "u_42", "ticket_id": "T-8891"},
    resource_map={
        "lookup_ticket": "crm:ticket",
        "fetch_customer": "crm:customer",
        "post_refund": "billing:refund",
    },
    action_map={"post_refund": "write"},
)
```

Without this, renaming `fetch_customer` to `get_account` silently changes which rules apply to it. With it, a rename is just a rename.

Unrecognised tool names fall back to `execute` — the most restricted action — so an unmapped tool fails closed rather than being treated as a harmless read.

---

## 7. Wrap a real agent

**LangChain:**

```python
from agenticpolicy import GuardedAgent

guarded = GuardedAgent(
    executor,
    policy=policy,
    context={"user_id": "u_42", "ticket_id": "T-8891"},
    resource_map={"lookup_ticket": "crm:ticket"},
)

result = guarded.invoke({"input": "What's the status of my ticket?"})
print(guarded.report())
```

**LlamaIndex:**

```python
from agenticpolicy import GuardedLlamaIndexAgent

guarded = GuardedLlamaIndexAgent.from_tools(tools, llm=llm, policy=policy)
guarded.chat("How many tickets are open?")
```

**LangGraph:**

```python
from agenticpolicy.integrations.langgraph_ import GuardedToolNode

graph.add_node("tools", GuardedToolNode(tools, policy=policy))
```

---

## 8. Choose how blocks surface

By default a blocked call returns a string the model reads as a tool observation:

```
[BLOCKED] Denied by rule deny_2 (no PII writes)
```

The agent sees the refusal and usually finds a permitted path — which is what you want in a conversational agent. If you'd rather handle it in code:

```python
from agenticpolicy.exceptions import PolicyViolation, ApprovalRequired

guarded = GuardedAgent(executor, policy=policy, on_violation="raise")

try:
    guarded.invoke({"input": "Delete all the tickets"})
except ApprovalRequired as e:
    queue_for_review(e.tool_call)
except PolicyViolation as e:
    log.warning("blocked: %s", e.decision.reason)
```

---

## 9. Turn on the audit log

```python
from agenticpolicy import EventStore, PolicyEngine

store = EventStore("audit.db")
engine = PolicyEngine(policy, store=store)
guarded = GuardedAgent(executor, engine=engine, context={...})
```

Every decision is recorded — allowed and blocked alike.

```bash
agenticpolicy audit --db audit.db --blocked-only
agenticpolicy summary --db audit.db --hours 24
```

Pass one `engine` to several guards and they share both the audit trail and the budget counters.

---

## 10. Lock the policy down in CI

Save it and assert its behaviour, so a well-meaning widening shows up in review:

```python
policy.save("policy.json")
```

```bash
agenticpolicy test policy.json delete crm:ticket --expect deny
agenticpolicy test policy.json write github:main --expect deny
agenticpolicy test policy.json execute ci:deploy --expect needs_approval
```

Exit code 1 on mismatch, so a policy change that quietly permits deletes fails the build.

---

## Start from a pre-built policy

```python
from agenticpolicy import PrebuiltRules

policy = PrebuiltRules.least_privilege_support_bot()
policy.allow("read", ["kb:runbooks"])      # extend it
```

`agenticpolicy catalog` lists them all. They're starting points, not compliance guarantees — read what each one does before shipping it.

---

## Next

- Three runnable examples, no API keys: `python examples/support_bot.py`
- [Policy reference](policy_reference.md) — every option, and the evaluation order
- [Architecture](architecture.md) — how it works and what it doesn't cover
