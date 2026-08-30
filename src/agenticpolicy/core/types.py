"""Core data types for agenticpolicy.

Everything here is a plain dataclass with no third-party dependencies, so the
core of the library imports in milliseconds and works without pydantic,
langchain, or a database driver installed.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ActionType(str, Enum):
    """What an agent is trying to do to a resource."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ANY = "*"

    @classmethod
    def coerce(cls, value: ActionType | str) -> ActionType:
        """Accept an ActionType or a string, case-insensitively.

        Raises:
            ValueError: with the list of valid actions, rather than the opaque
                message ``ActionType`` would raise on its own.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            valid = ", ".join(repr(a.value) for a in cls)
            raise ValueError(f"Unknown action {value!r}. Valid actions: {valid}") from None


class RuleType(str, Enum):
    """The kind of constraint a rule expresses."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "approve"
    PREVENT_EXFILTRATION = "prevent_exfil"
    REQUIRE_CONTEXT = "require_context"


class Effect(str, Enum):
    """Terminal outcome of an evaluation."""

    ALLOW = "allow"
    DENY = "deny"
    NEEDS_APPROVAL = "needs_approval"


#: Budget windows understood by ``require_approval(budget=...)``, in seconds.
BUDGET_WINDOWS: dict[str, int] = {
    "per_minute": 60,
    "per_hour": 3600,
    "per_day": 86_400,
    "per_task": 0,  # 0 == no time window; counted per context["task_id"]
}


@dataclass
class PolicyRule:
    """A single rule inside a :class:`~agenticpolicy.core.policy.Policy`."""

    action: ActionType
    resources: list[str]
    rule_type: RuleType
    conditions: dict[str, Any] | None = None
    budget: dict[str, int] | None = None
    limits: dict[str, int] | None = None
    description: str | None = None
    id: str = field(default_factory=_new_id)

    def __post_init__(self) -> None:
        self.action = ActionType.coerce(self.action)
        if not self.resources:
            raise ValueError(
                f"Rule {self.id} has an empty resource list. "
                'Use ["*"] to match every resource explicitly.'
            )
        if self.budget:
            unknown = set(self.budget) - set(BUDGET_WINDOWS)
            if unknown:
                valid = ", ".join(sorted(BUDGET_WINDOWS))
                raise ValueError(
                    f"Unknown budget window(s) {sorted(unknown)}. Valid windows: {valid}"
                )
            for window, limit in self.budget.items():
                if not isinstance(limit, int) or limit < 0:
                    raise ValueError(
                        f"Budget {window!r} must be a non-negative integer, got {limit!r}"
                    )

    def __str__(self) -> str:
        return f"{self.rule_type.value}({self.action.value}, {self.resources})"


@dataclass
class ToolCall:
    """A single tool invocation to be checked against a policy."""

    agent_id: str
    tool_name: str
    resource: str
    action: ActionType
    args: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    output: str | None = None
    id: str = field(default_factory=_new_id)
    timestamp: datetime = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        self.action = ActionType.coerce(self.action)

    def redacted_args(
        self, keys: tuple[str, ...] = ("password", "token", "api_key", "secret")
    ) -> dict[str, Any]:
        """Args with obviously-sensitive values masked, for safe logging."""
        out: dict[str, Any] = {}
        for k, v in self.args.items():
            out[k] = "***" if any(s in k.lower() for s in keys) else v
        return out


@dataclass
class PolicyDecision:
    """The result of evaluating a :class:`ToolCall` against a policy."""

    allowed: bool
    reason: str
    effect: Effect = Effect.DENY
    rule_applied: str | None = None
    requires_approval: bool = False
    risk_score: float = 0.0
    findings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls, reason: str, rule_applied: str | None = None, **kw: Any) -> PolicyDecision:
        return cls(
            allowed=True, reason=reason, effect=Effect.ALLOW, rule_applied=rule_applied, **kw
        )

    @classmethod
    def deny(cls, reason: str, rule_applied: str | None = None, **kw: Any) -> PolicyDecision:
        return cls(
            allowed=False, reason=reason, effect=Effect.DENY, rule_applied=rule_applied, **kw
        )

    @classmethod
    def approval(cls, reason: str, rule_applied: str | None = None, **kw: Any) -> PolicyDecision:
        return cls(
            allowed=False,
            reason=reason,
            effect=Effect.NEEDS_APPROVAL,
            requires_approval=True,
            rule_applied=rule_applied,
            **kw,
        )

    def __bool__(self) -> bool:
        return self.allowed

    def __str__(self) -> str:
        return f"{self.effect.value.upper()}: {self.reason}"


@dataclass
class PolicyEvent:
    """An audit record: one decision about one tool call."""

    agent_id: str
    tool_name: str
    resource: str
    action: str
    allowed: bool
    effect: str
    reason: str
    risk_score: float = 0.0
    rule_applied: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    id: str = field(default_factory=_new_id)
    timestamp: datetime = field(default_factory=_utcnow)

    @classmethod
    def from_decision(cls, call: ToolCall, decision: PolicyDecision) -> PolicyEvent:
        return cls(
            agent_id=call.agent_id,
            tool_name=call.tool_name,
            resource=call.resource,
            action=call.action.value,
            allowed=decision.allowed,
            effect=decision.effect.value,
            reason=decision.reason,
            risk_score=decision.risk_score,
            rule_applied=decision.rule_applied,
            context=dict(call.context),
            findings=list(decision.findings),
            timestamp=call.timestamp,
        )
