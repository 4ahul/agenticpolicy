"""The policy evaluation engine.

Evaluation happens in two phases, because some checks can only run before a
tool executes and others only after:

**Pre-call** (:meth:`PolicyEngine.evaluate`)
    Required context → deny rules → allow rules → conditions → outbound
    argument scan → approval gates → budgets. Nothing has executed yet, so a
    denial here means the tool never runs.

**Post-call** (:meth:`PolicyEngine.check_output`)
    Scans what the tool returned for blocked patterns and size limits. The
    original plan folded this into a single pass, but output does not exist
    before the call — a one-phase design can only ever check arguments.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from agenticpolicy.core.matching import check_conditions, matches_resource
from agenticpolicy.core.policy import Policy
from agenticpolicy.core.types import (
    BUDGET_WINDOWS,
    ActionType,
    PolicyDecision,
    PolicyEvent,
    PolicyRule,
    ToolCall,
)

__all__ = ["PolicyEngine", "BudgetTracker", "scan_text"]


@lru_cache(maxsize=512)
def _compile_block(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


def scan_text(text: str, patterns: list[str]) -> list[str]:
    """Return a description of each blocked pattern found in ``text``.

    Findings name the pattern and the number of matches, never the matched
    values themselves — an audit log that quotes the SSN it caught is its own
    data leak.
    """
    findings: list[str] = []
    for pattern in patterns:
        try:
            hits = _compile_block(pattern).findall(text)
        except re.error:
            # A malformed user-supplied pattern must not take down the agent.
            findings.append(f"invalid pattern skipped: {pattern!r}")
            continue
        if hits:
            findings.append(f"matched blocked pattern {pattern!r} ({len(hits)}x)")
    return findings


def redact_text(text: str, patterns: list[str], mask: str = "[REDACTED]") -> str:
    """Replace every blocked-pattern match in ``text`` with ``mask``."""
    for pattern in patterns:
        try:
            text = _compile_block(pattern).sub(mask, text)
        except re.error:
            continue
    return text


class BudgetTracker:
    """Sliding-window call counter backing ``budget={"per_hour": N}``.

    Timestamps are kept per (agent, rule, window) and trimmed on read, so a
    budget of "10 per hour" means ten calls in any rolling hour — not ten calls
    since the process started, which is what a plain counter would give.

    ``per_task`` has no time window; it counts against ``context["task_id"]``
    and resets when the agent moves to a new task.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._events: dict[tuple[str, str, str], deque[float]] = defaultdict(deque)
        self._task_counts: dict[tuple[str, str, str], int] = defaultdict(int)

    def _trim(self, key: tuple[str, str, str], window_seconds: int) -> deque[float]:
        bucket = self._events[key]
        cutoff = self._clock() - window_seconds
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        return bucket

    def usage(self, agent_id: str, rule_id: str, window: str, task_id: str = "") -> int:
        """How many calls have been recorded in the current window."""
        if window == "per_task":
            return self._task_counts[(agent_id, rule_id, task_id)]
        return len(self._trim((agent_id, rule_id, window), BUDGET_WINDOWS[window]))

    def would_exceed(
        self, agent_id: str, rule: PolicyRule, task_id: str = ""
    ) -> tuple[bool, str | None]:
        """True if recording one more call would breach any of the rule's windows."""
        if not rule.budget:
            return False, None
        for window, limit in rule.budget.items():
            used = self.usage(agent_id, rule.id, window, task_id)
            if used + 1 > limit:
                return True, f"budget {window}={limit} exhausted for rule {rule.id} (used {used})"
        return False, None

    def record(self, agent_id: str, rule: PolicyRule, task_id: str = "") -> None:
        """Count one successful call against every window on the rule."""
        if not rule.budget:
            return
        now = self._clock()
        for window in rule.budget:
            if window == "per_task":
                self._task_counts[(agent_id, rule.id, task_id)] += 1
            else:
                self._events[(agent_id, rule.id, window)].append(now)

    def reset(self, agent_id: str | None = None) -> None:
        """Clear tracked usage, for one agent or all of them."""
        if agent_id is None:
            self._events.clear()
            self._task_counts.clear()
            return
        for key in [k for k in self._events if k[0] == agent_id]:
            del self._events[key]
        for key in [k for k in self._task_counts if k[0] == agent_id]:
            del self._task_counts[key]


class PolicyEngine:
    """Evaluates tool calls against a :class:`Policy`.

    Args:
        policy: The rules to enforce.
        store: Optional audit store; anything with a ``log_event(PolicyEvent)``
            method works, including
            :class:`~agenticpolicy.audit.store.EventStore`.
        clock: Injectable monotonic clock, so budget windows are testable
            without sleeping.

    The engine holds the mutable runtime state (budget counters), while the
    policy stays immutable — share one policy across agents, give each agent
    its own engine.
    """

    #: Cap on evaluated-but-not-yet-charged calls held in memory. A call is
    #: normally discarded within microseconds, on :meth:`commit` or
    #: :meth:`discard`; the cap only matters for a caller that uses the engine
    #: directly and does neither, where an unbounded dict would be a slow leak
    #: in a long-running process.
    MAX_PENDING = 4096

    def __init__(
        self,
        policy: Policy,
        *,
        store: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy
        self.store = store
        self.budgets = BudgetTracker(clock=clock)
        self._pending_budget: OrderedDict[str, list[PolicyRule]] = OrderedDict()

    # ------------------------------------------------------------ pre-call

    async def evaluate(self, tool_call: ToolCall) -> PolicyDecision:
        """Decide whether ``tool_call`` may execute. Async for symmetry with
        agent frameworks; :meth:`evaluate_sync` is the blocking equivalent."""
        return self.evaluate_sync(tool_call)

    def evaluate_sync(self, tool_call: ToolCall) -> PolicyDecision:
        """Synchronous evaluation. All checks are CPU-bound, so this is the
        real implementation and :meth:`evaluate` simply delegates to it."""
        decision = self._evaluate(tool_call)
        self._log(tool_call, decision)
        return decision

    def _evaluate(self, tool_call: ToolCall) -> PolicyDecision:
        sources = (tool_call.context, tool_call.args)

        # 1. Required context — an agent that cannot identify itself gets nothing.
        missing = [f for f in self.policy.required_context if f not in tool_call.context]
        if missing:
            return PolicyDecision.deny(
                f"Missing required context field(s): {', '.join(missing)}",
                metadata={"missing_context": missing},
            )

        # 2. Deny rules, fail-fast. Deny always beats allow.
        for rule in self.policy.deny_rules:
            if self.policy.matches_rule(rule, tool_call.action, tool_call.resource, *sources):
                return PolicyDecision.deny(
                    f"Denied by rule {rule.id}"
                    + (f" ({rule.description})" if rule.description else ""),
                    rule_applied=rule.id,
                    risk_score=1.0,
                )

        # 3. Allow rules — default-deny, so one must match.
        matched: PolicyRule | None = None
        near_misses: list[str] = []
        for rule in self.policy.allow_rules:
            if rule.action is not ActionType.ANY and rule.action is not tool_call.action:
                continue
            if not any(matches_resource(p, tool_call.resource) for p in rule.resources):
                continue
            passed, why = check_conditions(rule.conditions, *sources)
            if passed:
                matched = rule
                break
            if why:
                near_misses.append(f"{rule.id}: {why}")

        if matched is None:
            reason = f"No allow rule permits {tool_call.action.value} on {tool_call.resource!r}"
            if near_misses:
                reason += f" (closest: {near_misses[0]})"
            return PolicyDecision.deny(
                reason, metadata={"near_misses": near_misses}, risk_score=0.6
            )

        # 4. Outbound argument scan — the payload an exfiltrating agent sends out.
        exfil = self.policy.exfil_limits
        if exfil and exfil.get("scan_args"):
            findings = scan_text(_stringify(tool_call.args), exfil["block_patterns"])
            if findings and not exfil.get("redact"):
                return PolicyDecision.deny(
                    "Blocked: tool arguments contain sensitive data",
                    rule_applied=matched.id,
                    risk_score=0.95,
                    findings=findings,
                )

        # 5. Approval gates and budgets, on the same rules.
        for rule in self.policy.approve_rules:
            if not self.policy.matches_rule(rule, tool_call.action, tool_call.resource, *sources):
                continue

            task_id = str(tool_call.context.get("task_id", ""))
            exceeded, why = self.budgets.would_exceed(tool_call.agent_id, rule, task_id)
            if exceeded:
                return PolicyDecision.deny(
                    f"Rate limit exceeded: {why}",
                    rule_applied=rule.id,
                    risk_score=0.5,
                    metadata={"budget": rule.budget},
                )

            approval_exempt = bool(rule.limits and rule.limits.get("approval_exempt"))
            if not approval_exempt:
                return PolicyDecision.approval(
                    f"Requires human approval (rule {rule.id})"
                    + (f": {rule.description}" if rule.description else ""),
                    rule_applied=rule.id,
                    risk_score=0.4,
                    metadata={"budget": rule.budget} if rule.budget else {},
                )

        # 6. Allowed. Note which budgeted rules to charge once the call runs.
        pending = [
            rule
            for rule in self.policy.approve_rules
            if rule.budget
            and self.policy.matches_rule(rule, tool_call.action, tool_call.resource, *sources)
        ]
        if pending:
            self._pending_budget[tool_call.id] = pending
            while len(self._pending_budget) > self.MAX_PENDING:
                # Oldest first. A call this stale was abandoned without a
                # commit, so dropping it forfeits the charge rather than
                # growing forever.
                self._pending_budget.popitem(last=False)
        return PolicyDecision.allow(
            f"Allowed by rule {matched.id}",
            rule_applied=matched.id,
        )

    def commit(self, tool_call: ToolCall) -> None:
        """Charge budgets for a call that actually executed.

        Kept separate from :meth:`evaluate` so a call that is evaluated but
        abandoned — the agent changed its mind, an upstream error fired — does
        not consume the user's rate limit.
        """
        task_id = str(tool_call.context.get("task_id", ""))
        for rule in self._pending_budget.pop(tool_call.id, []):
            self.budgets.record(tool_call.agent_id, rule, task_id)

    def discard(self, tool_call: ToolCall) -> None:
        """Forget a call that was allowed but never executed.

        Idempotent, and a no-op after :meth:`commit`. Guards call this in a
        ``finally`` so a tool that raises releases its reservation instead of
        leaving it pinned in memory for the life of the process.
        """
        self._pending_budget.pop(tool_call.id, None)

    # ----------------------------------------------------------- post-call

    def check_output(self, tool_call: ToolCall, output: Any) -> tuple[PolicyDecision, Any]:
        """Scan a tool's return value before it reaches the model.

        Returns the decision and the output to actually use — identical to the
        input unless the policy is in ``redact=True`` mode, in which case
        matches are masked and the call is allowed through.
        """
        exfil = self.policy.exfil_limits
        if not exfil or not exfil.get("scan_output"):
            return PolicyDecision.allow("No output policy configured"), output

        text = _stringify(output)
        size_kb = len(text.encode("utf-8")) / 1024
        max_kb = exfil["max_output_kb"]
        if size_kb > max_kb:
            decision = PolicyDecision.deny(
                f"Output too large: {size_kb:.1f}KB exceeds the {max_kb}KB limit",
                risk_score=0.9,
                metadata={"size_kb": round(size_kb, 2), "max_output_kb": max_kb},
            )
            self._log(tool_call, decision)
            return decision, None

        findings = scan_text(text, exfil["block_patterns"])
        if findings:
            if exfil.get("redact"):
                decision = PolicyDecision.allow(
                    f"Output redacted: {len(findings)} sensitive pattern(s) masked",
                    findings=findings,
                    risk_score=0.5,
                )
                self._log(tool_call, decision)
                return decision, redact_text(text, exfil["block_patterns"])
            decision = PolicyDecision.deny(
                "Blocked: tool output contains sensitive data",
                risk_score=0.95,
                findings=findings,
            )
            self._log(tool_call, decision)
            return decision, None

        return PolicyDecision.allow("Output passed exfiltration checks"), output

    # -------------------------------------------------------------- helpers

    def _log(self, tool_call: ToolCall, decision: PolicyDecision) -> None:
        if self.store is None:
            return
        try:
            self.store.log_event(PolicyEvent.from_decision(tool_call, decision))
        except Exception:  # pragma: no cover - auditing must never break the agent
            pass

    def dry_run(self, tool_call: ToolCall) -> PolicyDecision:
        """Evaluate without logging or reserving budget — for testing policies."""
        return self._evaluate(tool_call)


def _stringify(value: Any) -> str:
    """Flatten any tool payload to text for pattern scanning."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(f"{k}={_stringify(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_stringify(v) for v in value)
    return str(value)
