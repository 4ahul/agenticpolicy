"""Pre-built policies for common agent shapes.

Each factory returns a fresh :class:`Policy` you can extend::

    policy = PrebuiltRules.least_privilege_support_bot()
    policy.allow("read", ["kb:runbooks"])

They are starting points, not compliance guarantees — read what each one does
before shipping it.
"""

from __future__ import annotations

from agenticpolicy.core.policy import DEFAULT_BLOCK_PATTERNS, Policy

__all__ = ["PrebuiltRules"]


class PrebuiltRules:
    """A library of ready-made policies."""

    @staticmethod
    def no_data_exfiltration(max_output_kb: int = 100, redact: bool = False) -> Policy:
        """Read-only access with aggressive outbound data scanning.

        Blocks SSNs, credit cards, emails, AWS keys, private keys and secret
        assignments in both tool arguments and tool output.
        """
        policy = Policy(
            agent_id="exfil_guarded_agent",
            description="Read-only, with sensitive-data scanning in both directions",
        )
        policy.allow("read", ["*"], description="read anything")
        policy.deny("write", ["*"], description="no writes")
        policy.deny("delete", ["*"], description="no deletes")
        policy.deny("execute", ["*"], description="no code execution")
        policy.prevent_exfiltration(
            max_output_kb=max_output_kb,
            block_patterns=list(DEFAULT_BLOCK_PATTERNS.values()),
            redact=redact,
        )
        return policy

    @staticmethod
    def least_privilege_support_bot() -> Policy:
        """A support agent that can read tickets and docs, and update ticket status only.

        Customer and payment records stay off-limits for writes, deletes are
        blocked outright, and every call must carry a ticket and user id so the
        audit log is traceable.
        """
        policy = Policy(
            agent_id="support_bot",
            description="Support agent: read tickets/docs, update ticket status only",
        )
        policy.require_context(["ticket_id", "user_id"])
        policy.allow("read", ["crm:ticket", "docs:*", "kb:*"], description="research the issue")
        policy.allow(
            "write",
            ["crm:ticket"],
            conditions={"field": "status"},
            description="status transitions only",
        )
        policy.deny(
            "write", ["crm:customer", "crm:payment", "billing:*"], description="no PII writes"
        )
        policy.deny("delete", ["*"], description="support never deletes")
        policy.deny("execute", ["*"], description="no code execution")
        policy.prevent_exfiltration(max_output_kb=50)
        return policy

    @staticmethod
    def code_agent_with_gates(deploys_per_day: int = 5) -> Policy:
        """A coding agent that opens PRs freely but needs sign-off to deploy.

        Direct writes to ``github:main`` are denied — the agent must go through
        a pull request — and CI deploys are both approval-gated and capped.
        """
        policy = Policy(
            agent_id="code_agent",
            description="Reads repos, opens PRs, cannot push to main, deploys need approval",
        )
        policy.allow("read", ["github:*", "docs:*"], description="read the codebase")
        policy.allow("write", ["github:pr", "github:branch"], description="open PRs and branches")
        # Approval gates only fire on calls that some allow rule already permits —
        # under default-deny, gating an action you never allowed is a no-op.
        policy.allow("execute", ["ci:deploy", "ci:release"], description="deploys, gated below")
        policy.deny("write", ["github:main", "github:master"], description="protected branches")
        policy.deny(
            "delete", ["github:repo", "github:branch"], description="no destructive git ops"
        )
        policy.require_approval(
            "execute",
            ["ci:deploy", "ci:release"],
            budget={"per_day": deploys_per_day},
            description="human sign-off for deploys",
        )
        return policy

    @staticmethod
    def data_analyst(max_output_kb: int = 50) -> Policy:
        """Read-only warehouse access with a hard cap on how much data can leave.

        The size limit is the real control here: an analyst agent rarely needs
        to return more than a summary, and a bulk ``SELECT *`` is the shape
        that data exfiltration actually takes.
        """
        policy = Policy(
            agent_id="data_analyst",
            description="Read-only analytics with bulk-export protection",
        )
        policy.allow("read", ["postgres:*", "redshift:*", "snowflake:*", "bigquery:*"])
        policy.deny("write", ["*"], description="analysts do not write")
        policy.deny("delete", ["*"], description="analysts do not delete")
        policy.prevent_exfiltration(max_output_kb=max_output_kb)
        return policy

    @staticmethod
    def rate_limited_agent(calls_per_hour: int = 100) -> Policy:
        """Any-action agent capped at a fixed call rate.

        Unlike an approval gate this never blocks for a human — calls run until
        the hourly budget is spent, then they are denied until the window
        rolls forward.
        """
        policy = Policy(
            agent_id="rate_limited_agent",
            description=f"Unrestricted actions, capped at {calls_per_hour} calls/hour",
        )
        policy.allow("*", ["*"])
        policy.rate_limit("*", ["*"], per_hour=calls_per_hour)
        return policy

    @staticmethod
    def read_only(resources: list[str] | None = None) -> Policy:
        """The simplest useful guardrail: reads allowed, everything else denied."""
        policy = Policy(agent_id="read_only_agent", description="Reads only")
        policy.allow("read", resources or ["*"])
        policy.deny("write", ["*"])
        policy.deny("delete", ["*"])
        policy.deny("execute", ["*"])
        return policy

    @staticmethod
    def human_in_the_loop(write_resources: list[str] | None = None) -> Policy:
        """Reads run freely; every write and delete waits for a human."""
        policy = Policy(
            agent_id="hitl_agent",
            description="Reads are free, mutations require approval",
        )
        policy.allow("read", ["*"])
        policy.allow("write", write_resources or ["*"])
        policy.allow("delete", write_resources or ["*"])
        policy.require_approval(
            "write", write_resources or ["*"], description="review before write"
        )
        policy.require_approval("delete", ["*"], description="review before delete")
        policy.deny("execute", ["*"])
        return policy

    @classmethod
    def catalog(cls) -> dict[str, str]:
        """Map of factory name to its one-line summary, for docs and the CLI."""
        out: dict[str, str] = {}
        for name in dir(cls):
            if name.startswith("_") or name == "catalog":
                continue
            fn = getattr(cls, name)
            doc = (fn.__doc__ or "").strip().splitlines()
            out[name] = doc[0] if doc else ""
        return out
