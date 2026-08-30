"""LlamaIndex integration.

Install with::

    pip install "agenticpolicy[llamaindex]"

LlamaIndex agents take a list of ``FunctionTool`` objects, so the guard applies
at the same place as with LangChain: replace each tool with a guarded copy
before handing them to the agent.
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

__all__ = ["GuardedLlamaIndexAgent", "guard_llamaindex_tool", "guard_llamaindex_tools"]


def _build_agent(agent_cls: Any, tools: list[Any], kwargs: dict[str, Any]) -> Any:
    """Construct a LlamaIndex agent across framework versions.

    0.12+ takes ``tools=`` on the constructor; earlier releases used the
    ``from_tools`` classmethod. Each form is tried in turn and the last error
    is re-raised, so a genuine mistake in ``agent_kwargs`` still surfaces
    instead of being swallowed as a version mismatch.
    """
    attempts: list[Callable[[], Any]] = [
        lambda: agent_cls(tools=tools, **kwargs),
        lambda: agent_cls(tools, **kwargs),
    ]
    builder = getattr(agent_cls, "from_tools", None)
    if builder is not None:
        attempts.insert(0, lambda: builder(tools, **kwargs))

    last: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last = exc
    raise TypeError(
        f"Could not construct {agent_cls.__name__} with the guarded tools. "
        f"Pass a prebuilt agent to GuardedLlamaIndexAgent(...) instead. Last error: {last}"
    ) from last


def _require_llamaindex() -> Any:
    try:
        from llama_index.core.tools import FunctionTool

        return FunctionTool
    except ImportError:
        raise IntegrationNotInstalled("LlamaIndex", "llamaindex") from None


def guard_llamaindex_tool(
    tool: Any,
    guard: ToolGuard,
    *,
    resource: str | None = None,
    action: ActionType | str | None = None,
) -> Any:
    """Return a guarded copy of a LlamaIndex ``FunctionTool``."""
    FunctionTool = _require_llamaindex()

    metadata = tool.metadata
    name = metadata.name
    if resource is not None:
        guard.resource_map[name] = resource
    if action is not None:
        guard.action_map[name] = ActionType.coerce(action)

    sync_fn = getattr(tool, "fn", None)
    async_fn = getattr(tool, "async_fn", None)
    if sync_fn is None and async_fn is None:
        # Older builds expose only __call__.
        sync_fn = tool.call

    return FunctionTool.from_defaults(
        fn=guard.wrap(sync_fn, name=name) if sync_fn is not None else None,
        async_fn=guard.wrap(async_fn, name=name) if async_fn is not None else None,
        name=name,
        description=metadata.description,
        fn_schema=getattr(metadata, "fn_schema", None),
        return_direct=getattr(metadata, "return_direct", False),
    )


def guard_llamaindex_tools(tools: Sequence[Any], guard: ToolGuard) -> list[Any]:
    """Guard a list of LlamaIndex tools with one shared engine."""
    return [guard_llamaindex_tool(t, guard) for t in tools]


class GuardedLlamaIndexAgent:
    """Wraps a LlamaIndex agent so every tool call is policy-checked.

    Because LlamaIndex agents bind their tools at construction time, this class
    guards the tools you pass in and builds the agent for you::

        guarded = GuardedLlamaIndexAgent.from_tools(
            tools, llm=llm, policy=PrebuiltRules.data_analyst()
        )
        response = guarded.chat("How many open tickets are there?")

    If you already have an agent instance, pass it directly — its ``tools`` are
    guarded and the agent is rebuilt where the version allows it.
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
        _require_llamaindex()
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
        self.agent = agent

    @classmethod
    def from_tools(
        cls,
        tools: Sequence[Any],
        *,
        policy: Policy | None = None,
        engine: PolicyEngine | None = None,
        context: dict[str, Any] | None = None,
        resource_map: dict[str, str] | None = None,
        action_map: dict[str, ActionType | str] | None = None,
        on_violation: str = "block",
        agent_cls: Any | None = None,
        **agent_kwargs: Any,
    ) -> GuardedLlamaIndexAgent:
        """Guard ``tools`` and build a LlamaIndex agent around them.

        ``agent_cls`` defaults to ``ReActAgent``; any extra keyword arguments
        (``llm``, ``verbose``, ``system_prompt``, …) pass straight through.
        """
        _require_llamaindex()
        guard = ToolGuard(
            policy,
            engine=engine,
            context=context,
            resource_map=resource_map,
            action_map=action_map,
            on_violation=on_violation,
        )
        guarded_tools = guard_llamaindex_tools(tools, guard)

        if agent_cls is None:
            from llama_index.core.agent import ReActAgent

            agent_cls = ReActAgent

        # LlamaIndex 0.12 dropped ``from_tools`` in favour of a keyword
        # constructor. Try the modern form first, then the legacy builder, so
        # one adapter spans both without pinning the framework version.
        agent = _build_agent(agent_cls, guarded_tools, agent_kwargs)
        return cls(agent, _guard=guard)

    # ------------------------------------------------------------ delegation

    def chat(self, message: str, **kwargs: Any) -> Any:
        """Send a message to the agent.

        LlamaIndex 0.12 replaced ``chat()`` with the workflow-style ``run()``,
        so whichever the installed version offers is used. Both return the
        framework's own response object untouched.
        """
        for method in ("chat", "run"):
            fn = getattr(self.agent, method, None)
            if callable(fn):
                return fn(message, **kwargs)
        raise AttributeError(
            f"{type(self.agent).__name__} exposes neither chat() nor run(); "
            "call the agent directly via .agent"
        )

    async def achat(self, message: str, **kwargs: Any) -> Any:
        """Async counterpart of :meth:`chat`.

        ``AgentWorkflow.run`` returns an awaitable handler, so the result is
        awaited when it is one and returned as-is otherwise.
        """
        for method in ("achat", "arun", "run"):
            fn = getattr(self.agent, method, None)
            if callable(fn):
                result = fn(message, **kwargs)
                return await result if inspect.isawaitable(result) else result
        raise AttributeError(f"{type(self.agent).__name__} exposes no async chat entry point")

    def query(self, message: str, **kwargs: Any) -> Any:
        return self.agent.query(message, **kwargs)

    def invoke(self, input: dict[str, Any] | str, **kwargs: Any) -> Any:
        """LangChain-style entry point, for code that treats agents uniformly."""
        message = input["input"] if isinstance(input, dict) else input
        return self.chat(message, **kwargs)

    @property
    def blocked(self) -> list[Any]:
        return self.guard.blocked

    def report(self) -> str:
        return self.guard.report()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.agent, name)
