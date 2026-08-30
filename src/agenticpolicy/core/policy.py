"""The Policy DSL — a fluent builder for agent guardrails."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agenticpolicy.core.matching import check_conditions, matches_resource
from agenticpolicy.core.types import ActionType, PolicyRule, RuleType
from agenticpolicy.exceptions import PolicyConfigError

#: Patterns for data that should not leave the agent's boundary. Used as the
#: default when ``prevent_exfiltration`` is called with no explicit patterns.
DEFAULT_BLOCK_PATTERNS: dict[str, str] = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d[ -]?){13,16}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "aws_access_key": r"\bAKIA[0-9A-Z]{16}\b",
    "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "secret_assignment": r"(?i)\b(?:password|passwd|api[_-]?key|secret|bearer)\b\s*[:=]\s*\S+",
}


class Policy:
    """A set of rules describing what an agent may and may not do.

    Every mutator returns ``self``, so rules chain::

        policy = (
            Policy(agent_id="support_bot")
            .allow("read", ["crm:*"])
            .deny("delete", ["*"])
            .require_approval("write", ["crm:customer"])
        )

    Rules are evaluated by :class:`~agenticpolicy.core.engine.PolicyEngine`.
    The policy object itself holds no runtime state, so one policy can safely
    back several engines or agents.
    """

    def __init__(self, agent_id: str = "default_agent", *, description: str | None = None):
        self.agent_id = agent_id
        self.description = description
        self.allow_rules: list[PolicyRule] = []
        self.deny_rules: list[PolicyRule] = []
        self.approve_rules: list[PolicyRule] = []
        self.exfil_limits: dict[str, Any] | None = None
        self.required_context: list[str] = []
        self._rule_counter = 0

    # ------------------------------------------------------------------ DSL

    def allow(
        self,
        action: str | ActionType,
        resources: list[str],
        conditions: dict[str, Any] | None = None,
        *,
        description: str | None = None,
    ) -> Policy:
        """Permit ``action`` on ``resources``, optionally only when ``conditions`` hold.

        A tool call is denied unless some allow rule matches it, so this is the
        positive half of a default-deny model.
        """
        self.allow_rules.append(
            self._rule(
                RuleType.ALLOW, action, resources, conditions=conditions, description=description
            )
        )
        return self

    def deny(
        self,
        action: str | ActionType,
        resources: list[str],
        conditions: dict[str, Any] | None = None,
        *,
        description: str | None = None,
    ) -> Policy:
        """Forbid ``action`` on ``resources``. Deny always beats allow."""
        self.deny_rules.append(
            self._rule(
                RuleType.DENY, action, resources, conditions=conditions, description=description
            )
        )
        return self

    def require_approval(
        self,
        action: str | ActionType,
        resources: list[str] | None = None,
        budget: dict[str, int] | None = None,
        conditions: dict[str, Any] | None = None,
        *,
        description: str | None = None,
    ) -> Policy:
        """Hold matching calls for a human, and/or cap how often they may run.

        ``budget`` accepts ``per_minute``, ``per_hour``, ``per_day`` and
        ``per_task`` — for example ``{"per_day": 5}``. A rule with a budget but
        no other purpose acts as a pure rate limit: calls pass until the budget
        is spent.
        """
        self.approve_rules.append(
            self._rule(
                RuleType.REQUIRE_APPROVAL,
                action,
                resources or ["*"],
                conditions=conditions,
                budget=budget,
                description=description,
            )
        )
        return self

    def rate_limit(
        self,
        action: str | ActionType,
        resources: list[str] | None = None,
        *,
        per_minute: int | None = None,
        per_hour: int | None = None,
        per_day: int | None = None,
        per_task: int | None = None,
    ) -> Policy:
        """Cap call frequency without requiring human approval.

        Sugar over :meth:`require_approval` with ``budget`` set and approval
        suppressed — calls run freely until the budget is exhausted, then they
        are denied.
        """
        budget = {
            k: v
            for k, v in (
                ("per_minute", per_minute),
                ("per_hour", per_hour),
                ("per_day", per_day),
                ("per_task", per_task),
            )
            if v is not None
        }
        if not budget:
            raise PolicyConfigError(
                "rate_limit() needs at least one of per_minute, per_hour, per_day, per_task"
            )
        rule = self._rule(
            RuleType.REQUIRE_APPROVAL,
            action,
            resources or ["*"],
            budget=budget,
            description="rate limit",
        )
        rule.conditions = rule.conditions or None
        rule.limits = {"approval_exempt": 1}
        self.approve_rules.append(rule)
        return self

    def prevent_exfiltration(
        self,
        max_output_kb: int | None = None,
        block_patterns: list[str] | None = None,
        *,
        scan_args: bool = True,
        scan_output: bool = True,
        redact: bool = False,
    ) -> Policy:
        """Block sensitive data from leaving through tool args or tool output.

        Args:
            max_output_kb: Reject outputs larger than this. Defaults to 100.
            block_patterns: Regexes to reject. Defaults to
                :data:`DEFAULT_BLOCK_PATTERNS` (SSNs, cards, emails, AWS keys,
                private keys, secret assignments).
            scan_args: Also scan outbound tool arguments, which is where an
                agent tricked into exfiltration usually puts the payload.
            scan_output: Scan data coming back from the tool.
            redact: Mask matches and let the call through, instead of blocking.
        """
        patterns = (
            block_patterns if block_patterns is not None else list(DEFAULT_BLOCK_PATTERNS.values())
        )
        self.exfil_limits = {
            "max_output_kb": max_output_kb if max_output_kb is not None else 100,
            "block_patterns": patterns,
            "scan_args": scan_args,
            "scan_output": scan_output,
            "redact": redact,
        }
        return self

    def require_context(self, fields: list[str]) -> Policy:
        """Require these keys in every tool call's ``context``.

        Useful for forcing traceability — an agent that cannot say which user
        and which ticket it is acting for gets no tool access at all.
        """
        for f in fields:
            if f not in self.required_context:
                self.required_context.append(f)
        return self

    # ------------------------------------------------------------- helpers

    def _rule(
        self,
        rule_type: RuleType,
        action: str | ActionType,
        resources: list[str],
        *,
        conditions: dict[str, Any] | None = None,
        budget: dict[str, int] | None = None,
        description: str | None = None,
    ) -> PolicyRule:
        if isinstance(resources, str):
            raise PolicyConfigError(
                f"resources must be a list, got the string {resources!r}. "
                f'Did you mean resources=["{resources}"]?'
            )
        prefix = {
            RuleType.ALLOW: "allow",
            RuleType.DENY: "deny",
            RuleType.REQUIRE_APPROVAL: "approve",
        }.get(rule_type, "rule")
        rule = PolicyRule(
            action=ActionType.coerce(action),
            resources=list(resources),
            rule_type=rule_type,
            conditions=conditions,
            budget=budget,
            description=description,
            id=f"{prefix}_{self._rule_counter}",
        )
        self._rule_counter += 1
        return rule

    @property
    def rules(self) -> list[PolicyRule]:
        """Every rule in the policy, in declaration order."""
        return [*self.deny_rules, *self.allow_rules, *self.approve_rules]

    def matches_rule(
        self, rule: PolicyRule, action: ActionType, resource: str, *sources: dict[str, Any]
    ) -> bool:
        """True if ``rule`` applies to this action/resource/context combination."""
        if rule.action is not ActionType.ANY and rule.action is not action:
            return False
        if not any(matches_resource(p, resource) for p in rule.resources):
            return False
        passed, _ = check_conditions(rule.conditions, *sources)
        return passed

    def merge(self, other: Policy) -> Policy:
        """Return a new policy combining both rule sets.

        Deny rules from either side apply, and required-context fields union —
        merging never loosens a restriction.
        """
        merged = Policy(self.agent_id, description=self.description)
        for rule in (*self.rules, *other.rules):
            bucket = {
                RuleType.ALLOW: merged.allow_rules,
                RuleType.DENY: merged.deny_rules,
                RuleType.REQUIRE_APPROVAL: merged.approve_rules,
            }[rule.rule_type]
            clone = PolicyRule(
                action=rule.action,
                resources=list(rule.resources),
                rule_type=rule.rule_type,
                conditions=rule.conditions,
                budget=rule.budget,
                limits=rule.limits,
                description=rule.description,
                id=f"{rule.id}_m{merged._rule_counter}",
            )
            merged._rule_counter += 1
            bucket.append(clone)
        merged.required_context = sorted(set(self.required_context) | set(other.required_context))
        if self.exfil_limits and other.exfil_limits:
            merged.exfil_limits = {
                "max_output_kb": min(
                    self.exfil_limits["max_output_kb"], other.exfil_limits["max_output_kb"]
                ),
                "block_patterns": list(
                    dict.fromkeys(
                        [
                            *self.exfil_limits["block_patterns"],
                            *other.exfil_limits["block_patterns"],
                        ]
                    )
                ),
                "scan_args": self.exfil_limits["scan_args"] or other.exfil_limits["scan_args"],
                "scan_output": self.exfil_limits["scan_output"]
                or other.exfil_limits["scan_output"],
                "redact": self.exfil_limits["redact"] and other.exfil_limits["redact"],
            }
        else:
            merged.exfil_limits = self.exfil_limits or other.exfil_limits
        return merged

    # ---------------------------------------------------------- validation

    def validate(self) -> list[str]:
        """Return warnings about rules that cannot do what they appear to do.

        Evaluation is default-deny, which makes one mistake easy and silent: an
        approval gate or rate limit on an action that no allow rule permits
        never fires, because the call is already denied before the gate is
        reached. The policy *looks* like it gates deploys; in fact it forbids
        them outright.

        Warnings are advisory — a policy with warnings still evaluates. Call
        this in tests, or run ``agenticpolicy explain`` to see them.
        """
        warnings: list[str] = []

        for rule in self.approve_rules:
            covered = any(
                (allow.action is ActionType.ANY or allow.action is rule.action)
                and any(
                    matches_resource(allow_pattern, resource.replace("*", "x"))
                    or matches_resource(resource, allow_pattern.replace("*", "x"))
                    for allow_pattern in allow.resources
                    for resource in rule.resources
                )
                for allow in self.allow_rules
            )
            if not covered:
                kind = (
                    "rate limit" if (rule.limits or {}).get("approval_exempt") else "approval gate"
                )
                warnings.append(
                    f"{rule.id}: {kind} on {rule.action.value} {rule.resources} is unreachable — "
                    f"no allow rule permits it, so those calls are denied outright. "
                    f'Add policy.allow("{rule.action.value}", {rule.resources}) if you meant to gate them.'
                )

        for allow in self.allow_rules:
            for deny in self.deny_rules:
                if deny.conditions or allow.conditions:
                    continue
                if deny.action is not ActionType.ANY and deny.action is not allow.action:
                    continue
                if set(deny.resources) >= set(allow.resources):
                    warnings.append(
                        f"{allow.id}: allow {allow.action.value} {allow.resources} is fully "
                        f"shadowed by deny rule {deny.id} — deny always wins."
                    )

        if self.exfil_limits and self.exfil_limits["max_output_kb"] == 0:
            warnings.append(
                "exfiltration: max_output_kb=0 blocks every non-empty tool output. "
                "Use a positive limit, or omit prevent_exfiltration()."
            )

        return warnings

    # --------------------------------------------------- (de)serialization

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict, suitable for JSON or version control."""

        def dump(rule: PolicyRule) -> dict[str, Any]:
            d: dict[str, Any] = {
                "id": rule.id,
                "action": rule.action.value,
                "resources": rule.resources,
            }
            for key in ("conditions", "budget", "limits", "description"):
                value = getattr(rule, key)
                if value:
                    d[key] = value
            return d

        return {
            "agent_id": self.agent_id,
            "description": self.description,
            "allow": [dump(r) for r in self.allow_rules],
            "deny": [dump(r) for r in self.deny_rules],
            "require_approval": [dump(r) for r in self.approve_rules],
            "required_context": self.required_context,
            "exfiltration": self.exfil_limits,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Policy:
        """Rebuild a policy from :meth:`to_dict` output."""
        policy = cls(data.get("agent_id", "default_agent"), description=data.get("description"))
        for r in data.get("deny", []):
            policy.deny(
                r["action"], r["resources"], r.get("conditions"), description=r.get("description")
            )
        for r in data.get("allow", []):
            policy.allow(
                r["action"], r["resources"], r.get("conditions"), description=r.get("description")
            )
        for r in data.get("require_approval", []):
            policy.require_approval(
                r["action"],
                r["resources"],
                r.get("budget"),
                r.get("conditions"),
                description=r.get("description"),
            )
            if r.get("limits"):
                policy.approve_rules[-1].limits = r["limits"]
        if data.get("required_context"):
            policy.require_context(list(data["required_context"]))
        exfil = data.get("exfiltration")
        if exfil:
            policy.prevent_exfiltration(
                max_output_kb=exfil.get("max_output_kb"),
                block_patterns=exfil.get("block_patterns"),
                scan_args=exfil.get("scan_args", True),
                scan_output=exfil.get("scan_output", True),
                redact=exfil.get("redact", False),
            )
        return policy

    def save(self, path: str | Path) -> None:
        """Write the policy to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Policy:
        """Read a policy back from a JSON file written by :meth:`save`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # -------------------------------------------------------------- dunder

    def __str__(self) -> str:
        return (
            f"Policy(agent={self.agent_id!r}, allow={len(self.allow_rules)}, "
            f"deny={len(self.deny_rules)}, approve={len(self.approve_rules)})"
        )

    __repr__ = __str__

    def explain(self) -> str:
        """A human-readable summary, handy in code review and PR descriptions."""
        lines = [f"Policy for agent {self.agent_id!r}"]
        if self.description:
            lines.append(f"  {self.description}")
        if self.required_context:
            lines.append(f"  requires context: {', '.join(self.required_context)}")
        for label, rules in (
            ("DENY", self.deny_rules),
            ("ALLOW", self.allow_rules),
            ("APPROVE", self.approve_rules),
        ):
            for rule in rules:
                bits = [f"  [{label}] {rule.action.value} on {', '.join(rule.resources)}"]
                if rule.conditions:
                    bits.append(f"if {rule.conditions}")
                if rule.budget:
                    bits.append(f"budget {rule.budget}")
                lines.append(" ".join(bits))
        if self.exfil_limits:
            lines.append(
                f"  [EXFIL] max {self.exfil_limits['max_output_kb']}KB, "
                f"{len(self.exfil_limits['block_patterns'])} blocked patterns"
            )
        warnings = self.validate()
        if warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend(f"  ! {w}" for w in warnings)
        return "\n".join(lines)
