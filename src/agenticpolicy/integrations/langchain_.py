"""LangChain integration.

Install with::

    pip install "agenticpolicy[langchain]"

Three entry points:

:func:`guard_tool` / :func:`guard_tools`
    Wrap LangChain ``BaseTool`` objects, returning new tools with the same
    names and schemas. Use these when you build the tool list yourself.

:meth:`GuardedAgent.create`
    Guards the tools and then builds the agent around them. This is the path
    for LangChain 1.x, where ``create_agent`` compiles tools into a graph that
    cannot be swapped afterwards.

:class:`GuardedAgent`
    Wraps an existing agent that exposes ``.tools`` — the classic
    ``AgentExecutor`` shape — guarding all of them at once.

All three replace the tools rather than monkeypatching them. The original design
patched ``tool.invoke`` in place inside a ``try/finally``: that mutates objects
the caller still holds, leaks the patch if the process dies mid-run, and
double-patches under concurrency. Building new tool objects avoids all three.
"""

from __future__ import annotations

import inspect
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

    sync_fn: Callable[..., Any] | None = getattr(tool, "func", None)
    async_fn: Callable[..., Any] | None = getattr(tool, "coroutine", None)
    if sync_fn is None and async_fn is None:
        # A BaseTool subclass implementing _run directly exposes neither.
        sync_fn = tool.run

    guarded_sync = guard.wrap(_relay(sync_fn, name), name=name) if sync_fn else None
    guarded_async = guard.wrap(_relay(async_fn, name), name=name) if async_fn else None

    return StructuredTool.from_function(
        func=guarded_sync,
        coroutine=guarded_async,
        name=name,
        description=tool.description,
        args_schema=getattr(tool, "args_schema", None),
        return_direct=getattr(tool, "return_direct", False),
    )


def _relay(fn: Callable[..., Any], name: str) -> Callable[..., Any]:
    """A pass-through with ``fn``'s signature, so the guard binds real names.

    ``ToolGuard`` maps positional arguments to parameter names before evaluating
    conditions. A relay declared ``(*args, **kwargs)`` would hide those names
    and any condition written against an argument would silently never match,
    so the signature is copied across.
    """
    if inspect.iscoroutinefunction(fn):

        async def async_relay(*args: Any, **kwargs: Any) -> Any:
            return await fn(*args, **kwargs)

        relay: Callable[..., Any] = async_relay
    else:

        def sync_relay(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        relay = sync_relay

    relay.__name__ = name
    relay.__doc__ = getattr(fn, "__doc__", None)
    try:
        relay.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
    except (TypeError, ValueError):  # pragma: no cover - builtins
        pass
    return relay


def guard_tools(tools: Sequence[Any], guard: ToolGuard) -> list[Any]:
    """Guard a list of LangChain tools with one shared policy engine."""
    return [guard_tool(t, guard) for t in tools]


class GuardedAgent:
    """A LangChain agent with every tool policy-enforced.

    Args:
        agent: An agent exposing ``.tools`` and ``.invoke`` — the classic
            ``AgentExecutor`` shape. LangChain 1.x agents built by
            ``create_agent`` bind their tools into a compiled graph and expose
            no ``.tools``; use :meth:`create` for those.
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
        _guard: ToolGuard | None = None,
    ) -> None:
        _require_langchain()
        self.guard = _guard or ToolGuard(
            policy,
            engine=engine,
            context=context,
            resource_map=resource_map,
            action_map=action_map,
            on_violation=on_violation,
        )
        self.policy = self.guard.policy
        self.engine = self.guard.engine

        if _guard is not None:
            # Tools were guarded before the agent was built (see .create).
            self.agent = agent
            return

        tools = list(getattr(agent, "tools", []) or [])
        if not tools:
            raise ValueError(
                "The agent exposes no .tools, so there is nothing to guard. "
                "LangChain 1.x agents from create_agent() bind tools into a "
                "compiled graph — use GuardedAgent.create(model=..., tools=..., "
                "policy=...) instead, or guard the tools with guard_tools() "
                "before you build the agent."
            )
        self.agent = self._rebuild(agent, guard_tools(tools, self.guard))

    @classmethod
    def create(
        cls,
        *,
        model: Any,
        tools: Sequence[Any],
        policy: Policy | None = None,
        engine: PolicyEngine | None = None,
        context: dict[str, Any] | None = None,
        resource_map: dict[str, str] | None = None,
        action_map: dict[str, ActionType | str] | None = None,
        on_violation: str = "block",
        agent_factory: Callable[..., Any] | None = None,
        **agent_kwargs: Any,
    ) -> GuardedAgent:
        """Guard ``tools`` first, then build the agent around them.

        This is the supported path on LangChain 1.x, where ``create_agent``
        compiles the tools into a graph that cannot be swapped afterwards.
        Guarding before construction is also strictly safer: there is no window
        in which the agent holds unguarded tools.

        ``agent_factory`` defaults to ``langchain.agents.create_agent``; extra
        keyword arguments pass straight through to it::

            guarded = GuardedAgent.create(
                model=llm,
                tools=[lookup_ticket, close_ticket],
                policy=PrebuiltRules.least_privilege_support_bot(),
                context={"user_id": "u_42", "ticket_id": "t_991"},
            )
        """
        _require_langchain()
        guard = ToolGuard(
            policy,
            engine=engine,
            context=context,
            resource_map=resource_map,
            action_map=action_map,
            on_violation=on_violation,
        )
        guarded_tools = guard_tools(list(tools), guard)

        factory = agent_factory
        if factory is None:
            try:
                from langchain.agents import create_agent
            except ImportError:
                raise IntegrationNotInstalled("LangChain agents", "langchain") from None
            factory = create_agent

        agent = factory(model=model, tools=guarded_tools, **agent_kwargs)
        return cls(agent, _guard=guard)

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
