# Policy Reference

Complete reference for the policy DSL, evaluation order, and every option.

---

## Evaluation order

Checks run in a fixed order, and the first failure wins. The order matters: it determines which reason you see when several rules could have blocked a call.

**Before the tool runs** (`PolicyEngine.evaluate`):

1. **Required context** — every field in `require_context` must be present.
2. **Deny rules** — first match wins, always. Deny beats allow.
3. **Allow rules** — one must match, including its conditions. Default-deny.
4. **Outbound argument scan** — tool arguments checked against blocked patterns.
5. **Approval gates** — matching calls held for a human.
6. **Budgets** — matching calls checked against their rolling windows.

**After the tool runs** (`PolicyEngine.check_output`):

7. **Output size** — against `max_output_kb`.
8. **Output pattern scan** — against `block_patterns`.

Two consequences worth internalizing:

- **An approval gate on an action no allow rule permits never fires.** Step 3 denies the call before step 5 is reached. `policy.validate()` flags this.
- **Output checks can only happen after execution.** The tool has already run when its output is rejected; what's prevented is the data reaching the model and, through it, the user. For side effects you must prevent, use deny rules or approval gates.

---

## Actions

| Action | Meaning |
|---|---|
| `read` | Retrieve data |
| `write` | Create or modify data |
| `delete` | Remove data |
| `execute` | Run code, deploy, trigger a job |
| `*` | Any action |

Case-insensitive: `"READ"` and `"read"` are the same. An unknown action raises `ValueError` listing the valid ones.

---

## Resources

Resources are `provider:type` strings — `salesforce:lead`, `github:repo`, `postgres:users`.

### Wildcards

| Pattern | Matches | Does not match |
|---|---|---|
| `salesforce:lead` | `salesforce:lead` | `salesforce:contact` |
| `salesforce:*` | `salesforce:lead`, `salesforce:contact` | `salesforce:lead:notes`, `hubspot:lead` |
| `salesforce:**` | `salesforce:lead`, `salesforce:lead:notes` | `hubspot:lead` |
| `*:lead` | `salesforce:lead`, `hubspot:lead` | `salesforce:contact` |
| `*` | everything | — |

`*` matches within one segment; `**` crosses `:`. A pattern that is exactly `*` means everything.

### Matching rules

- **Case-insensitive.** `Salesforce:Lead` matches `salesforce:*`.
- **Metacharacters are literal.** `crm.ticket` matches only the resource literally named `crm.ticket`, not `crm:ticket`. (A naive `replace("*", ".*")` implementation gets this wrong and silently widens rules.)
- **Compiled patterns are cached**, so evaluation does no regex compilation on the hot path.

---

## Rules

### `allow(action, resources, conditions=None, description=None)`

Permit an action. Evaluation is default-deny, so nothing is permitted without a matching allow rule.

```python
policy.allow("read", ["crm:ticket", "kb:*"])
policy.allow("write", ["crm:ticket"], conditions={"field": "status"})
```

### `deny(action, resources, conditions=None, description=None)`

Forbid an action. Deny always wins, whatever else the policy says. Use it for resources that must never be touched — a deny rule is the one control that cannot be widened by adding another allow.

```python
policy.deny("delete", ["*"])
policy.deny("write", ["github:main", "github:master"])
```

The `description` shows up in block messages and the audit log, so future-you knows why:

```python
policy.deny("write", ["github:main"], description="protected branch")
# -> "[BLOCKED] Denied by rule deny_0 (protected branch)"
```

### `require_approval(action, resources=None, budget=None, conditions=None)`

Hold matching calls for a human, and optionally cap how often they may run.

```python
policy.allow("execute", ["ci:deploy"])                   # required first
policy.require_approval("execute", ["ci:deploy"], budget={"per_day": 5})
```

Blocked calls return `PolicyDecision` with `requires_approval=True` and `effect=NEEDS_APPROVAL`, distinct from a flat denial, so your UI can route them to a review queue.

### `rate_limit(action, resources=None, *, per_minute, per_hour, per_day, per_task)`

A budget with no human in the loop. Calls run freely until the budget is spent, then they're denied until the window rolls forward.

```python
policy.rate_limit("execute", ["ci:*"], per_hour=10)
policy.rate_limit("*", ["*"], per_day=1000)
```

### `prevent_exfiltration(...)`

```python
policy.prevent_exfiltration(
    max_output_kb=50,
    block_patterns=[r"\b\d{3}-\d{2}-\d{4}\b"],
    scan_args=True,
    scan_output=True,
    redact=False,
)
```

| Argument | Default | Effect |
|---|---|---|
| `max_output_kb` | `100` | Reject outputs above this size |
| `block_patterns` | see below | Regexes to reject. `[]` means size-only |
| `scan_args` | `True` | Scan outbound tool arguments |
| `scan_output` | `True` | Scan returned data |
| `redact` | `False` | Mask matches and allow, instead of blocking |

Default patterns (`DEFAULT_BLOCK_PATTERNS`): SSN, credit card, email, AWS access key, PEM private key, secret assignment (`password=`, `api_key:`, `bearer …`).

Only one exfiltration config per policy — a second call replaces the first.

### `require_context(fields)`

Require these keys in every call's `context`. An agent that can't say who it's acting for gets no tool access at all.

```python
policy.require_context(["user_id", "ticket_id"])
```

Supply them once via `ToolGuard(context={...})` or `GuardedAgent(context={...})`.

---

## Conditions

Conditions are checked against the tool call's `context` first, then its `args`. **Context wins** — trusted execution context overrides model-supplied arguments, so an agent can't talk its way past a condition by passing a different value.

Three shapes:

```python
conditions={"status": "open"}                  # equality
conditions={"status": ["open", "pending"]}     # membership
conditions={"amount": {"lte": 1000}}           # operators
```

### Operators

| Operator | Meaning |
|---|---|
| `eq`, `ne` | Equal / not equal |
| `in`, `not_in` | Membership |
| `gt`, `gte`, `lt`, `lte` | Ordering |
| `contains` | Substring or collection membership |
| `matches` | Regex search |
| `exists` | Field presence (`True`/`False`) |

Dotted paths work: `{"user.role": "admin"}`.

Conditions **fail closed**: a missing field or a type mismatch (comparing `"abc"` to `1000`) fails the condition rather than raising.

---

## Introspection

### `policy.validate() -> list[str]`

Warnings about rules that can't do what they appear to do:

- Approval gates and rate limits on actions no allow rule permits (unreachable).
- Allow rules fully shadowed by a deny rule.
- `max_output_kb=0`, which blocks every non-empty output.

Advisory — a policy with warnings still evaluates. Assert `policy.validate() == []` in tests.

### `policy.explain() -> str`

Human-readable summary including warnings. Good in PR descriptions.

### `policy.merge(other) -> Policy`

Combine two policies. Deny rules from both apply, required context unions, and the *stricter* exfiltration limit wins. Merging never loosens a restriction. Neither original is modified.

---

## Serialization

```python
policy.save("policy.json")
policy = Policy.load("policy.json")

data = policy.to_dict()        # JSON-serializable
policy = Policy.from_dict(data)
```

Round-tripping preserves behaviour, not rule ids. Check policies into version control so widening shows up in review.

---

## Testing policies

In pytest:

```python
from agenticpolicy import PolicyEngine
from agenticpolicy.core.types import ToolCall, ActionType

engine = PolicyEngine(policy)
decision = engine.dry_run(ToolCall(
    agent_id="bot", tool_name="delete_user", resource="crm:customer",
    action=ActionType.DELETE, args={}, context={"user_id": "u1"},
))
assert not decision.allowed
```

`dry_run` evaluates without logging or reserving budget.

In CI:

```bash
agenticpolicy test policy.json write github:main --expect deny
agenticpolicy test policy.json execute ci:deploy --expect needs_approval
agenticpolicy test policy.json read crm:ticket --context '{"user_id":"u1"}' --expect allow
```

Exit code 1 on mismatch.

---

## Complete examples

### Support bot

```python
policy = Policy(agent_id="support_bot_v1")
policy.require_context(["user_id", "ticket_id"])
policy.allow("read", ["crm:ticket", "docs:internal", "kb:*"])
policy.allow("write", ["crm:ticket"], conditions={"field": "status"})
policy.deny("write", ["crm:customer", "billing:*"], description="no PII writes")
policy.deny("delete", ["*"], description="support never deletes")
policy.deny("execute", ["*"])
policy.prevent_exfiltration(max_output_kb=50)
```

### Code agent

```python
policy = Policy(agent_id="code_agent")
policy.allow("read", ["github:*", "docs:*"])
policy.allow("write", ["github:pr", "github:branch"])
policy.allow("execute", ["ci:deploy"])                  # so the gate can fire
policy.deny("write", ["github:main"], description="protected branch")
policy.deny("delete", ["github:repo", "github:branch"])
policy.require_approval("execute", ["ci:deploy"], budget={"per_day": 5})
```

### Data analyst

```python
policy = Policy(agent_id="analyst")
policy.allow("read", ["postgres:*", "redshift:*"])
policy.deny("write", ["*"])
policy.deny("delete", ["*"])
policy.prevent_exfiltration(max_output_kb=50)   # the real control
```

### Financial agent with a value threshold

```python
policy = Policy(agent_id="finance_bot")
policy.require_context(["user_id", "approver_role"])
policy.allow("read", ["ledger:*"])
policy.allow("write", ["ledger:transaction"], conditions={"amount": {"lte": 1000}})
policy.require_approval("write", ["ledger:transaction"],
                        conditions={"amount": {"gt": 1000}},
                        description="large transactions need sign-off")
policy.deny("delete", ["ledger:*"], description="ledger is append-only")
```
