"""Shared fixtures."""

from __future__ import annotations

import pytest

from agenticpolicy import EventStore, Policy, PolicyEngine, PrebuiltRules
from agenticpolicy.core.types import ActionType, ToolCall


class FakeClock:
    """A manually advanced monotonic clock, so budget windows are testable
    without sleeping through a real hour."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def support_policy() -> Policy:
    """The policy from the design docs: support bot, least privilege."""
    policy = Policy(agent_id="support_bot_v1")
    policy.allow("read", ["salesforce:lead", "postgres:tickets"])
    policy.allow("write", ["salesforce:lead"])
    policy.deny("delete", ["*"])
    policy.require_approval("write", ["salesforce:lead"])
    policy.prevent_exfiltration(max_output_kb=10)
    policy.require_context(["user_id", "ticket_id"])
    return policy


@pytest.fixture
def engine(support_policy: Policy) -> PolicyEngine:
    return PolicyEngine(support_policy)


@pytest.fixture
def store() -> EventStore:
    with EventStore(":memory:") as s:
        yield s


def make_call(
    action: str = "read",
    resource: str = "salesforce:lead",
    *,
    agent_id: str = "support_bot_v1",
    tool_name: str = "salesforce_read",
    args: dict | None = None,
    context: dict | None = None,
) -> ToolCall:
    """Build a ToolCall with sensible defaults for the support fixture."""
    return ToolCall(
        agent_id=agent_id,
        tool_name=tool_name,
        resource=resource,
        action=ActionType.coerce(action),
        args=args if args is not None else {},
        context=context if context is not None else {"user_id": "u1", "ticket_id": "t1"},
    )


@pytest.fixture
def call_factory():
    return make_call


@pytest.fixture
def analyst_policy() -> Policy:
    return PrebuiltRules.data_analyst(max_output_kb=1)
