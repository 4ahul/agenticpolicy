"""Exceptions raised by agenticpolicy."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from agenticpolicy.core.types import PolicyDecision, ToolCall


class AgenticPolicyError(Exception):
    """Base class for every error this library raises."""


class PolicyViolation(AgenticPolicyError):
    """A tool call was blocked by policy.

    Raised by guarded agents running in ``on_violation="raise"`` mode. The
    offending call and the decision that blocked it are attached so callers can
    log or re-check them.
    """

    def __init__(self, decision: PolicyDecision, tool_call: ToolCall) -> None:
        self.decision = decision
        self.tool_call = tool_call
        super().__init__(
            f"Blocked {tool_call.action.value} on {tool_call.resource} "
            f"via tool {tool_call.tool_name!r}: {decision.reason}"
        )


class ApprovalRequired(PolicyViolation):
    """A tool call needs human approval before it can run."""


class PolicyConfigError(AgenticPolicyError):
    """A policy was constructed with invalid or contradictory rules."""


class IntegrationNotInstalled(AgenticPolicyError):
    """An optional framework integration was used without its dependency."""

    def __init__(self, framework: str, extra: str) -> None:
        super().__init__(
            f"The {framework} integration requires extra dependencies. "
            f'Install them with: pip install "agenticpolicy[{extra}]"'
        )
