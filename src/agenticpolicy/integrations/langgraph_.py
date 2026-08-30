"""LangGraph integration.

Install with::

    pip install "agenticpolicy[langgraph]"

LangGraph has no middleware hook — the original design sketched
``graph.apply_middleware(guard)``, but that API does not exist. The real
interception point is the tool node, so this module offers:

:class:`GuardedToolNode`
    A drop-in replacement for ``ToolNode`` whose tools are policy-checked.

:func:`guard_node`
    Wraps any node function, checking policy against state before it runs —
    for graphs whose side effects live in ordinary nodes rather than tools.

Both read execution context out of the graph state, so ``require_context``
fields (``user_id``, ``task_id``, …) flow through the graph automatically.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from agenticpolicy.core.engine import PolicyEngine
from agenticpolicy.core.policy import Policy
from agenticpolicy.core.types import ActionType, ToolCall
from agenticpolicy.exceptions import IntegrationNotInstalled
from agenticpolicy.integrations.base import ToolGuard

__all__ = ["GuardedToolNode", "guard_node", "GuardMiddleware"]

#: State keys lifted into policy context automatically.
CONTEXT_KEYS = ("user_id", "task_id", "ticket_id", "request_id", "session_id", "thread_id")


def _require_langgraph() -> Any:
    try:
        from langgraph.prebuilt import ToolNode

        return ToolNode
    except ImportError:
        raise IntegrationNotInstalled("LangGraph", "langgraph") from None


def _context_from_state(state: Any, extra_keys: Sequence[str] = ()) -> dict[str, Any]:
    """Pull policy context out of graph state.

    Looks at the top level and inside a ``context`` sub-dict, so both common
    state layouts work without configuration.
    """
    if not isinstance(state, dict):
        return {}
    raw_nested = state.get("context")
    nested: dict[str, Any] = raw_nested if isinstance(raw_nested, dict) else {}
    context: dict[str, Any] = {}
    for key in (*CONTEXT_KEYS, *extra_keys):
        if key in nested:
            context[key] = nested[key]
        elif key in state:
            context[key] = state[key]
    return context


class GuardedToolNode:
    """A policy-enforcing ``ToolNode``.

    Use it exactly where you would use ``ToolNode``::

        from agenticpolicy.integrations.langgraph_ import GuardedToolNode

        graph.add_node("tools", GuardedToolNode(tools, policy=policy))

    Each tool is wrapped before the node is built, so blocked calls return an
    explanatory ``ToolMessage`` the model can read — the graph keeps running
    and the agent can choose a permitted path instead of crashing.
    """

    def __init__(
        self,
        tools: Sequence[Any],
        policy: Policy | None = None,
        *,
        engine: PolicyEngine | None = None,
        context: dict[str, Any] | None = None,
        resource_map: dict[str, str] | None = None,
        action_map: dict[str, ActionType | str] | None = None,
        on_violation: str = "block",
        context_keys: Sequence[str] = (),
        **tool_node_kwargs: Any,
    ) -> None:
        ToolNode = _require_langgraph()
        from agenticpolicy.integrations.langchain_ import guard_tools

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
        self.context_keys = tuple(context_keys)
        self._node = ToolNode(guard_tools(list(tools), self.guard), **tool_node_kwargs)

    def __call__(self, state: Any, *args: Any, **kwargs: Any) -> Any:
        # Refresh per-invocation context from state before the tools run.
        self.guard.context.update(_context_from_state(state, self.context_keys))
        return self._node.invoke(state, *args, **kwargs)

    def invoke(self, state: Any, *args: Any, **kwargs: Any) -> Any:
        return self(state, *args, **kwargs)

    async def ainvoke(self, state: Any, *args: Any, **kwargs: Any) -> Any:
        self.guard.context.update(_context_from_state(state, self.context_keys))
        return await self._node.ainvoke(state, *args, **kwargs)

    @property
    def blocked(self) -> list[Any]:
        return self.guard.blocked

    def report(self) -> str:
        return self.guard.report()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._node, name)


def guard_node(
    policy: Policy | None = None,
    *,
    resource: str,
    action: ActionType | str,
    engine: PolicyEngine | None = None,
    on_violation: str = "block",
    blocked_key: str = "policy_block",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that policy-checks a plain LangGraph node.

    For graphs where the risky operation is a node rather than a tool::

        @guard_node(policy, resource="postgres:users", action="delete")
        def purge_users(state):
            db.execute("DELETE FROM users WHERE ...")
            return state

    When a call is blocked the node returns the state unchanged with the reason
    under ``blocked_key``, so a conditional edge can route to a human-review
    branch instead of failing the run.
    """
    guard = ToolGuard(policy, engine=engine, on_violation=on_violation)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        node_name = getattr(fn, "__name__", "node")

        def wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
            call = ToolCall(
                agent_id=guard.policy.agent_id,
                tool_name=node_name,
                resource=resource,
                action=ActionType.coerce(action),
                args={"state_keys": sorted(state)} if isinstance(state, dict) else {},
                context=_context_from_state(state),
            )
            decision = guard.engine.evaluate_sync(call)
            guard.decisions.append((call, decision))
            if not decision.allowed:
                if on_violation == "raise":
                    guard._refuse(call, decision)
                if isinstance(state, dict):
                    return {**state, blocked_key: decision.reason}
                return state
            guard.engine.commit(call)
            return fn(state, *args, **kwargs)

        wrapper.__name__ = node_name
        wrapper.__doc__ = fn.__doc__
        wrapper.guard = guard  # type: ignore[attr-defined]
        return wrapper

    return decorator


class GuardMiddleware:
    """Deprecated alias kept for the name used in the original design docs.

    LangGraph exposes no middleware hook, so this simply constructs a
    :class:`GuardedToolNode`. Prefer that class directly.
    """

    def __new__(cls, tools: Sequence[Any], policy: Policy | None = None, **kwargs: Any) -> Any:
        import warnings

        warnings.warn(
            "GuardMiddleware is a compatibility shim; LangGraph has no middleware "
            "API. Use GuardedToolNode(tools, policy=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return GuardedToolNode(tools, policy, **kwargs)
