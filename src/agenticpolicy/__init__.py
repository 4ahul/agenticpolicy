"""agenticpolicy — safety guardrails for AI agents.

Add policy enforcement to an existing agent in a few lines::

    from agenticpolicy import Policy, GuardedAgent

    policy = Policy(agent_id="support_bot")
    policy.allow("read", ["crm:*"])
    policy.deny("delete", ["*"])

    guarded = GuardedAgent(agent, policy=policy)
    result = guarded.invoke({"input": "Look up ticket 4821"})

The core (``Policy``, ``PolicyEngine``, ``EventStore``) has no third-party
dependencies. Framework integrations are imported lazily, so importing this
package does not pull in LangChain.
"""

from __future__ import annotations

from typing import Any

from agenticpolicy.audit.store import EventStore
from agenticpolicy.core.engine import BudgetTracker, PolicyEngine
from agenticpolicy.core.policy import DEFAULT_BLOCK_PATTERNS, Policy
from agenticpolicy.core.rules import PrebuiltRules
from agenticpolicy.core.types import (
    ActionType,
    Effect,
    PolicyDecision,
    PolicyEvent,
    PolicyRule,
    RuleType,
    ToolCall,
)
from agenticpolicy.exceptions import (
    AgenticPolicyError,
    ApprovalRequired,
    IntegrationNotInstalled,
    PolicyConfigError,
    PolicyViolation,
)

__version__ = "0.1.0"

__all__ = [
    "ActionType",
    "AgenticPolicyError",
    "ApprovalRequired",
    "BudgetTracker",
    "DEFAULT_BLOCK_PATTERNS",
    "Effect",
    "EventStore",
    "GuardedAgent",
    "GuardedLlamaIndexAgent",
    "GuardedToolNode",
    "IntegrationNotInstalled",
    "Policy",
    "PolicyConfigError",
    "PolicyDecision",
    "PolicyEngine",
    "PolicyEvent",
    "PolicyRule",
    "PolicyViolation",
    "PrebuiltRules",
    "RuleType",
    "ToolCall",
    "__version__",
    "guard_tool",
]

# Integrations import their framework at call time, not import time, so
# `import agenticpolicy` stays fast and works with none of them installed.
_LAZY: dict[str, tuple[str, str]] = {
    "GuardedAgent": ("agenticpolicy.integrations.langchain_", "GuardedAgent"),
    "guard_tool": ("agenticpolicy.integrations.langchain_", "guard_tool"),
    "GuardedLlamaIndexAgent": ("agenticpolicy.integrations.llamaindex_", "GuardedLlamaIndexAgent"),
    "GuardedToolNode": ("agenticpolicy.integrations.langgraph_", "GuardedToolNode"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
