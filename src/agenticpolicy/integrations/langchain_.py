"""LangChain integration.

Install with::

    pip install "agenticpolicy[langchain]"

Two entry points:

:func:`guard_tool`
    Wraps one LangChain ``BaseTool``, returning a new tool with the same name
    and schema. Use it when you build the tool list yourself.

:class:`GuardedAgent`
    Wraps an ``AgentExecutor``, guarding all of its tools at once.

Both replace the tools rather than monkeypatching them. The original design
patched ``tool.invoke`` in place inside a ``try/finally``: that mutates objects
the caller still holds, leaks the patch if the process dies mid-run, and
double-patches under concurrency. Building new tool objects avoids all three.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from agenticpolicy.core.engine import PolicyEngine
from agenticpolicy.core.policy import Policy
from agenticpolicy.core.types import ActionType
from agenticpolicy.exceptions import IntegrationNotInstalled
from agenticpolicy.integrations.base import ToolGuard

__all__ = ["GuardedAgent", "guard_tool", "guard_tools"]


def _require_langchain() -> Any:
    try:
        from langchain_core.tools import BaseTool, StructuredTool  # noqa: F401

        return StructuredTool
    except ImportError:
        try:
            from langchain.tools import BaseTool, StructuredTool  # type: ignore # noqa: F401

            return StructuredTool
        except ImportError:
            raise IntegrationNotInstalled("LangChain", "langchain") from None


def guard_tool(
    tool: Any,
    guard: ToolGuard,
    *,
    resource: str | None = None,
    action: ActionType | str | None = None,
) -> Any:
    """Return a policy-enforcing copy of a LangChain tool.

    Name, description and argument schema are preserved, so the model sees no
    difference and prompts do not need rewriting. When a call is blocked the
    tool returns an explanatory string — the agent reads it as an observation
    and typically tries a permitted path instead.
    """
    StructuredTool = _require_langchain()

    name = tool.name
    if resource is not None:
        guard.resource_map[name] = resource
    if action is not None:
        guard.action_map[name] = ActionType.coerce(action)

    underlying: Callable[..., Any] = getattr(tool, "func", None) or tool.run

    def call(*args: Any, **kwargs: Any) -> Any:
        return underlying(*args, **kwargs)

    call.__name__ = name
    guarded_fn = guard.wrap(call, name=name)

    return StructuredTool.from_function(
        func=guarded_fn,
        name=name,
        description=tool.description,
        args_schema=getattr(tool, "args_schema", None),
        return_direct=getattr(tool, "return_direct", False),
    )


def guard_tools(tools: Sequence[Any], guard: ToolGuard) -> list[Any]:
    """Guard a list of LangChain tools with one shared policy engine."""
    return [guard_tool(t, guard) for t in tools]


class GuardedAgent:
    """A LangChain ``AgentExecutor`` with every tool policy-enforced.

    Args:
        agent: An ``AgentExecutor`` (or anything exposing ``.tools`` and
            ``.invoke``).
        policy: The policy to enforce.
        engine: Share an engine instead, when several agents should draw on
            one budget and one audit log.
        context: Baseline context merged into every tool call.
        resource_map / action_map: Explicit per-tool overrides. Strongly
            recommended in production — name inference is for prototyping.
        on_violation: ``"block"`` (default) feeds the refusal back to the model
            as an observation; ``"raise"`` raises ``PolicyViolation``.

    Example::

        policy = PrebuiltRules.least_privilege_support_bot()
        guarded = GuardedAgent(
            executor,
            policy=policy,
            context={"user_id": "u_42", "ticket_id": "t_991"},
            resource_map={"lookup_ticket": "crm:ticket"},
        )
        result = guarded.invoke({"input": "What is the status of my ticket?"})
        print(guarded.report())
    """

    def __init__(
        self,
        agent: Any,
        policy: Policy | None = None,
        *,
        engine: PolicyEngine | None = None,
        context: dict[str, Any] | None = None,
        resource_map: dict[str, str] | None = None,
        action_map: dict[str, ActionType | str] | None = None,
        on_violation: str = "block",
    ) -> None:
        _require_langchain()
        self.guard = ToolGuard(
            policy,
            engine=engine,
            context=context,
            resource_map=resource_map,
            action_map=action_map,
            on_violation=on_violation,
        )
        self.policy = self.guard.policy
        self.engine = self.guard.engine

        tools = list(getattr(agent, "tools", []) or [])
        if not tools:
            raise ValueError(
                "The agent exposes no .tools, so there is nothing to guard. "
                "Pass an AgentExecutor built with tools, or guard the tools "
                "directly with guard_tool() before constructing the agent."
            )
        self.agent = self._rebuild(agent, guard_tools(tools, self.guard))

    @staticmethod
    def _rebuild(agent: Any, guarded_tools: list[Any]) -> Any:
        """Produce an executor bound to the guarded tools.

        ``AgentExecutor`` is a pydantic model, so ``.copy(update=...)`` gives a
        new instance without touching the caller's object.
        """
        for attempt in ("model_copy", "copy"):
            copier = getattr(agent, attempt, None)
            if copier is None:
                continue
            try:
                return copier(update={"tools": guarded_tools})
            except Exception:  # pragma: no cover - version differences
                continue
        agent.tools = guarded_tools  # last resort
        return agent

    # ------------------------------------------------------------ delegation

    def invoke(self, input: dict[str, Any] | str, **kwargs: Any) -> Any:
        """Run the agent with guardrails active."""
        return self.agent.invoke(input, **kwargs)

    async def ainvoke(self, input: dict[str, Any] | str, **kwargs: Any) -> Any:
        """Async counterpart of :meth:`invoke`."""
        return await self.agent.ainvoke(input, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Legacy LangChain entry point."""
        return self.agent.run(*args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        return self.agent.stream(*args, **kwargs)

    # --------------------------------------------------------------- reports

    @property
    def blocked(self) -> list[Any]:
        """Calls refused during the run."""
        return self.guard.blocked

    def report(self) -> str:
        """Text summary of policy activity during the run."""
        return self.guard.report()

    def __getattr__(self, name: str) -> Any:
        # Anything not overridden falls through to the wrapped executor.
        return getattr(self.agent, name)
