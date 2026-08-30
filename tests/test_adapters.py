"""Framework adapter tests against real LangChain, LangGraph and LlamaIndex.

The suite in ``test_integrations.py`` covers the shared :class:`ToolGuard`
logic with no framework installed. This file covers the parts that only a real
framework can exercise: whether a guarded tool still satisfies the framework's
own contract, and whether a blocked call reaches the model as an observation it
can recover from.

Every class skips cleanly when its framework is absent, so the core suite still
runs on a bare install.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest

from agenticpolicy import Policy, PrebuiltRules
from agenticpolicy.exceptions import PolicyViolation
from agenticpolicy.integrations.base import ToolGuard


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - namespace edge cases
        return False


requires_langchain = pytest.mark.skipif(
    not _installed("langchain_core"), reason="langchain-core not installed"
)
requires_langchain_agents = pytest.mark.skipif(
    not _installed("langchain.agents"), reason="langchain not installed"
)
requires_langgraph = pytest.mark.skipif(
    not _installed("langgraph"), reason="langgraph not installed"
)
requires_llamaindex = pytest.mark.skipif(
    not _installed("llama_index.core"), reason="llama-index-core not installed"
)


def refund_policy() -> Policy:
    """Small refunds pass, large ones are denied — a condition on an argument.

    Conditions on arguments are the case an adapter can silently break: if the
    adapter passes arguments through as ``**kwargs`` without binding parameter
    names, the condition never matches and the rule looks enforced while doing
    nothing.
    """
    return (
        Policy("refund_bot")
        .allow("write", ["crm:refund"])
        .deny("write", ["crm:refund"], conditions={"amount": {"gt": 100}})
    )


def refund_guard() -> ToolGuard:
    return ToolGuard(
        refund_policy(),
        resource_map={"issue_refund": "crm:refund"},
        action_map={"issue_refund": "write"},
    )


# --------------------------------------------------------------------- LangChain


@requires_langchain
class TestLangChainTools:
    def test_metadata_and_schema_survive_guarding(self) -> None:
        from langchain_core.tools import StructuredTool

        from agenticpolicy.integrations.langchain_ import guard_tool

        def delete_record(record_id: str) -> str:
            """Delete a record."""
            return "deleted"

        tool = StructuredTool.from_function(
            func=delete_record, name="delete_record", description="Delete a record"
        )
        guarded = guard_tool(tool, ToolGuard(PrebuiltRules.read_only()))

        assert guarded.name == "delete_record"
        assert guarded.description == "Delete a record"
        # The model sees the same arguments, so no prompt needs rewriting.
        assert set(guarded.args) == set(tool.args)

    def test_blocked_tool_returns_observation_and_skips_side_effect(self) -> None:
        from langchain_core.tools import StructuredTool

        from agenticpolicy.integrations.langchain_ import guard_tool

        executed: list[str] = []

        def delete_record(record_id: str) -> str:
            executed.append(record_id)
            return "deleted"

        guarded = guard_tool(
            StructuredTool.from_function(func=delete_record, name="delete_record", description="d"),
            ToolGuard(PrebuiltRules.read_only()),
        )
        result = guarded.invoke({"record_id": "r1"})

        assert result.startswith("[BLOCKED]")
        assert executed == []

    def test_argument_conditions_are_evaluated(self) -> None:
        """The guard must see ``amount``, not an opaque kwargs blob."""
        from langchain_core.tools import StructuredTool

        from agenticpolicy.integrations.langchain_ import guard_tool

        def issue_refund(amount: int) -> str:
            return f"refunded {amount}"

        guarded = guard_tool(
            StructuredTool.from_function(func=issue_refund, name="issue_refund", description="r"),
            refund_guard(),
        )

        assert guarded.invoke({"amount": 50}) == "refunded 50"
        assert guarded.invoke({"amount": 1000}).startswith("[BLOCKED]")

    async def test_async_tools_keep_their_coroutine(self) -> None:
        """A tool defined only as a coroutine must stay awaitable.

        Wrapping it as a sync ``func`` would make ``ainvoke`` run the guard in a
        thread and return an un-awaited coroutine as the tool result.
        """
        from langchain_core.tools import StructuredTool

        from agenticpolicy.integrations.langchain_ import guard_tool

        async def issue_refund(amount: int) -> str:
            return f"refunded {amount}"

        tool = StructuredTool.from_function(
            coroutine=issue_refund, name="issue_refund", description="r"
        )
        guarded = guard_tool(tool, refund_guard())

        assert guarded.coroutine is not None
        assert await guarded.ainvoke({"amount": 50}) == "refunded 50"
        assert (await guarded.ainvoke({"amount": 1000})).startswith("[BLOCKED]")

    def test_guard_tools_shares_one_engine(self) -> None:
        from langchain_core.tools import StructuredTool

        from agenticpolicy.integrations.langchain_ import guard_tools

        guard = ToolGuard(PrebuiltRules.read_only())
        tools = guard_tools(
            [
                StructuredTool.from_function(func=lambda: "x", name="get_a", description="a"),
                StructuredTool.from_function(func=lambda: "y", name="delete_b", description="b"),
            ],
            guard,
        )
        for tool in tools:
            tool.invoke({})

        assert len(guard.decisions) == 2
        assert len(guard.blocked) == 1

    def test_raise_mode_propagates_through_the_tool(self) -> None:
        from langchain_core.tools import StructuredTool

        from agenticpolicy.integrations.langchain_ import guard_tool

        guarded = guard_tool(
            StructuredTool.from_function(func=lambda: "gone", name="delete_all", description="d"),
            ToolGuard(PrebuiltRules.read_only(), on_violation="raise"),
        )
        with pytest.raises(PolicyViolation):
            guarded.invoke({})


class ScriptedModel:
    """A chat model that replays a fixed list of messages.

    Defined lazily inside the test that needs it, because it must subclass a
    LangChain base class that may not be installed.
    """


def _scripted_model(messages: list[Any]) -> Any:
    """Build a minimal deterministic chat model over ``messages``.

    ``GenericFakeChatModel`` cannot be used here: it raises
    ``NotImplementedError`` from ``bind_tools``, which every agent calls.
    """
    from langchain_core.language_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult

    class _Scripted(BaseChatModel):
        script: list[Any] = []
        cursor: int = 0

        @property
        def _llm_type(self) -> str:
            return "scripted"

        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            # Tool-calling is scripted, so the schemas are irrelevant here.
            return self

        def _generate(
            self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
        ) -> Any:
            index = min(self.cursor, len(self.script) - 1)
            object.__setattr__(self, "cursor", self.cursor + 1)
            return ChatResult(generations=[ChatGeneration(message=self.script[index])])

    return _Scripted(script=messages)


@requires_langchain_agents
class TestGuardedAgentEndToEnd:
    """A full agent loop with a scripted model and no API key."""

    def test_blocked_call_reaches_the_model_as_an_observation(self) -> None:
        from langchain_core.messages import AIMessage
        from langchain_core.tools import StructuredTool

        from agenticpolicy.integrations.langchain_ import GuardedAgent

        executed: list[str] = []

        def delete_record(record_id: str) -> str:
            """Delete a record."""
            executed.append(record_id)
            return "deleted"

        model = _scripted_model(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "delete_record", "args": {"record_id": "r1"}, "id": "call_1"}
                    ],
                ),
                AIMessage(content="I am not permitted to delete records."),
            ]
        )
        guarded = GuardedAgent.create(
            model=model,
            tools=[
                StructuredTool.from_function(
                    func=delete_record, name="delete_record", description="Delete a record"
                )
            ],
            policy=PrebuiltRules.read_only(),
        )
        result = guarded.invoke({"messages": [("user", "delete record r1")]})

        # The side effect never happened...
        assert executed == []
        # ...the refusal came back as a tool observation...
        contents = [str(m.content) for m in result["messages"]]
        assert any("[BLOCKED]" in c for c in contents)
        # ...and the agent kept running instead of crashing.
        assert contents[-1] == "I am not permitted to delete records."
        assert len(guarded.blocked) == 1
        assert "delete_record" in guarded.report()

    def test_allowed_call_executes_normally(self) -> None:
        from langchain_core.messages import AIMessage
        from langchain_core.tools import StructuredTool

        from agenticpolicy.integrations.langchain_ import GuardedAgent

        def get_ticket(ticket_id: str) -> str:
            """Read a ticket."""
            return "status=open"

        model = _scripted_model(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "get_ticket", "args": {"ticket_id": "t1"}, "id": "call_1"}
                    ],
                ),
                AIMessage(content="The ticket is open."),
            ]
        )
        guarded = GuardedAgent.create(
            model=model,
            tools=[
                StructuredTool.from_function(
                    func=get_ticket, name="get_ticket", description="Read a ticket"
                )
            ],
            policy=PrebuiltRules.read_only(),
        )
        result = guarded.invoke({"messages": [("user", "status of t1?")]})

        assert any("status=open" in str(m.content) for m in result["messages"])
        assert guarded.blocked == []

    def test_context_flows_into_require_context(self) -> None:
        """Baseline context must satisfy ``require_context`` without threading
        it through every call site."""
        from langchain_core.messages import AIMessage
        from langchain_core.tools import StructuredTool

        from agenticpolicy.integrations.langchain_ import GuardedAgent

        def read_crm_ticket(ticket_id: str) -> str:
            """Read a ticket."""
            return "status=open"

        tools = [
            StructuredTool.from_function(
                func=read_crm_ticket, name="read_crm_ticket", description="Read a ticket"
            )
        ]
        script = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read_crm_ticket", "args": {"ticket_id": "t1"}, "id": "call_1"}
                ],
            ),
            AIMessage(content="done"),
        ]

        without = GuardedAgent.create(
            model=_scripted_model(list(script)),
            tools=tools,
            policy=PrebuiltRules.least_privilege_support_bot(),
        )
        without.invoke({"messages": [("user", "status of t1?")]})
        assert len(without.blocked) == 1
        assert "context" in without.blocked[0][1].reason.lower()

        with_context = GuardedAgent.create(
            model=_scripted_model(list(script)),
            tools=tools,
            policy=PrebuiltRules.least_privilege_support_bot(),
            context={"user_id": "u1", "ticket_id": "t1"},
        )
        result = with_context.invoke({"messages": [("user", "status of t1?")]})
        assert with_context.blocked == []
        assert any("status=open" in str(m.content) for m in result["messages"])

    def test_wrapping_a_toolless_agent_is_an_actionable_error(self) -> None:
        from agenticpolicy.integrations.langchain_ import GuardedAgent

        class Toolless:
            def invoke(self, _: Any) -> Any:  # pragma: no cover - never called
                return None

        with pytest.raises(ValueError, match="GuardedAgent.create"):
            GuardedAgent(Toolless(), PrebuiltRules.read_only())

    def test_audit_store_records_the_denial(self, store: Any) -> None:
        from langchain_core.messages import AIMessage
        from langchain_core.tools import StructuredTool

        from agenticpolicy import PolicyEngine
        from agenticpolicy.integrations.langchain_ import GuardedAgent

        model = _scripted_model(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "delete_record", "args": {"record_id": "r1"}, "id": "call_1"}
                    ],
                ),
                AIMessage(content="refused"),
            ]
        )
        guarded = GuardedAgent.create(
            model=model,
            tools=[
                StructuredTool.from_function(
                    func=lambda record_id: "deleted", name="delete_record", description="d"
                )
            ],
            engine=PolicyEngine(PrebuiltRules.read_only(), store=store),
        )
        guarded.invoke({"messages": [("user", "delete r1")]})

        events = store.blocked_events()
        assert len(events) == 1
        assert events[0].tool_name == "delete_record"
        assert events[0].action == "delete"


# --------------------------------------------------------------------- LangGraph


@requires_langgraph
class TestLangGraph:
    @staticmethod
    def _graph(node: Any) -> Any:
        """Compile a one-node graph. ``ToolNode`` requires a graph runtime, so
        it cannot be invoked standalone."""
        from langgraph.graph import END, MessagesState, StateGraph

        builder = StateGraph(MessagesState)
        builder.add_node("tools", node)
        builder.set_entry_point("tools")
        builder.add_edge("tools", END)
        return builder.compile()

    @staticmethod
    def _tool_call(name: str, args: dict[str, Any]) -> Any:
        from langchain_core.messages import AIMessage

        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call_1"}])

    def test_guarded_tool_node_blocks_and_returns_a_tool_message(self) -> None:
        from langchain_core.tools import StructuredTool

        from agenticpolicy.integrations.langgraph_ import GuardedToolNode

        executed: list[str] = []

        def delete_record(record_id: str) -> str:
            executed.append(record_id)
            return "deleted"

        node = GuardedToolNode(
            [
                StructuredTool.from_function(
                    func=delete_record, name="delete_record", description="d"
                )
            ],
            PrebuiltRules.read_only(),
        )
        out = self._graph(node).invoke(
            {"messages": [self._tool_call("delete_record", {"record_id": "r1"})]}
        )

        assert executed == []
        assert "[BLOCKED]" in str(out["messages"][-1].content)
        assert len(node.blocked) == 1

    def test_guarded_tool_node_allows_permitted_calls(self) -> None:
        from langchain_core.tools import StructuredTool

        from agenticpolicy.integrations.langgraph_ import GuardedToolNode

        node = GuardedToolNode(
            [
                StructuredTool.from_function(
                    func=lambda ticket_id: "status=open", name="get_ticket", description="d"
                )
            ],
            PrebuiltRules.read_only(),
        )
        out = self._graph(node).invoke(
            {"messages": [self._tool_call("get_ticket", {"ticket_id": "t1"})]}
        )

        assert str(out["messages"][-1].content) == "status=open"
        assert node.blocked == []

    def test_context_is_lifted_out_of_graph_state(self) -> None:
        """``require_context`` fields must flow from state, not be re-threaded
        into every tool call."""
        from langchain_core.tools import StructuredTool
        from langgraph.graph import END, MessagesState, StateGraph

        from agenticpolicy.integrations.langgraph_ import GuardedToolNode

        # Subclassing MessagesState rather than declaring a fresh TypedDict:
        # LangGraph resolves state annotations with get_type_hints, which cannot
        # see names local to this function.
        class State(MessagesState):
            user_id: str
            ticket_id: str

        node = GuardedToolNode(
            [
                StructuredTool.from_function(
                    func=lambda ticket_id: "status=open",
                    name="read_crm_ticket",
                    description="d",
                )
            ],
            PrebuiltRules.least_privilege_support_bot(),
        )
        builder = StateGraph(State)
        builder.add_node("tools", node)
        builder.set_entry_point("tools")
        builder.add_edge("tools", END)
        graph = builder.compile()

        out = graph.invoke(
            {
                "messages": [self._tool_call("read_crm_ticket", {"ticket_id": "t1"})],
                "user_id": "u1",
                "ticket_id": "t1",
            }
        )
        assert str(out["messages"][-1].content) == "status=open"
        assert node.blocked == []

    def test_guard_node_routes_instead_of_failing(self) -> None:
        from agenticpolicy.integrations.langgraph_ import guard_node

        purged: list[str] = []

        @guard_node(PrebuiltRules.read_only(), resource="postgres:users", action="delete")
        def purge_users(state: dict[str, Any]) -> dict[str, Any]:
            purged.append("done")
            return {**state, "purged": True}

        result = purge_users({"user_id": "u1"})

        assert purged == []
        assert "policy_block" in result
        assert result.get("purged") is None

    def test_guard_node_runs_permitted_work(self) -> None:
        from agenticpolicy.integrations.langgraph_ import guard_node

        @guard_node(PrebuiltRules.read_only(), resource="postgres:users", action="read")
        def load_users(state: dict[str, Any]) -> dict[str, Any]:
            return {**state, "users": ["ana"]}

        result = load_users({"user_id": "u1"})
        assert result["users"] == ["ana"]
        assert "policy_block" not in result

    def test_middleware_shim_warns_and_still_guards(self) -> None:
        from langchain_core.tools import StructuredTool

        from agenticpolicy.integrations.langgraph_ import GuardedToolNode, GuardMiddleware

        with pytest.warns(DeprecationWarning, match="GuardedToolNode"):
            node = GuardMiddleware(
                [
                    StructuredTool.from_function(
                        func=lambda: "gone", name="delete_all", description="d"
                    )
                ],
                PrebuiltRules.read_only(),
            )
        assert isinstance(node, GuardedToolNode)


# ------------------------------------------------------------------- LlamaIndex


@requires_llamaindex
class TestLlamaIndex:
    def test_metadata_survives_guarding(self) -> None:
        from llama_index.core.tools import FunctionTool

        from agenticpolicy.integrations.llamaindex_ import guard_llamaindex_tool

        tool = FunctionTool.from_defaults(
            fn=lambda record_id: "deleted",
            name="delete_record",
            description="Delete a record",
        )
        guarded = guard_llamaindex_tool(tool, ToolGuard(PrebuiltRules.read_only()))

        assert guarded.metadata.name == "delete_record"
        assert guarded.metadata.description == "Delete a record"

    def test_blocked_call_skips_side_effect(self) -> None:
        from llama_index.core.tools import FunctionTool

        from agenticpolicy.integrations.llamaindex_ import guard_llamaindex_tool

        executed: list[str] = []

        def delete_record(record_id: str) -> str:
            executed.append(record_id)
            return "deleted"

        guarded = guard_llamaindex_tool(
            FunctionTool.from_defaults(fn=delete_record, name="delete_record", description="d"),
            ToolGuard(PrebuiltRules.read_only()),
        )
        output = guarded.call(record_id="r1")

        assert executed == []
        assert "[BLOCKED]" in str(output)

    def test_argument_conditions_are_evaluated(self) -> None:
        from llama_index.core.tools import FunctionTool

        from agenticpolicy.integrations.llamaindex_ import guard_llamaindex_tool

        def issue_refund(amount: int) -> str:
            return f"refunded {amount}"

        guarded = guard_llamaindex_tool(
            FunctionTool.from_defaults(fn=issue_refund, name="issue_refund", description="r"),
            refund_guard(),
        )

        assert "refunded 50" in str(guarded.call(amount=50))
        assert "[BLOCKED]" in str(guarded.call(amount=1000))

    async def test_async_tool_stays_async(self) -> None:
        from llama_index.core.tools import FunctionTool

        from agenticpolicy.integrations.llamaindex_ import guard_llamaindex_tool

        async def issue_refund(amount: int) -> str:
            return f"refunded {amount}"

        guarded = guard_llamaindex_tool(
            FunctionTool.from_defaults(async_fn=issue_refund, name="issue_refund", description="r"),
            refund_guard(),
        )

        assert "refunded 50" in str(await guarded.acall(amount=50))
        assert "[BLOCKED]" in str(await guarded.acall(amount=1000))

    def test_guard_many_tools_shares_one_engine(self) -> None:
        from llama_index.core.tools import FunctionTool

        from agenticpolicy.integrations.llamaindex_ import guard_llamaindex_tools

        guard = ToolGuard(PrebuiltRules.read_only())
        tools = guard_llamaindex_tools(
            [
                FunctionTool.from_defaults(fn=lambda: "x", name="get_a", description="a"),
                FunctionTool.from_defaults(fn=lambda: "y", name="delete_b", description="b"),
            ],
            guard,
        )
        for tool in tools:
            tool.call()

        assert len(guard.decisions) == 2
        assert len(guard.blocked) == 1

    def test_agent_construction_spans_framework_versions(self) -> None:
        """``from_tools`` was removed in LlamaIndex 0.12; the adapter must build
        an agent either way, and say so clearly when it cannot."""
        from llama_index.core.tools import FunctionTool

        from agenticpolicy.integrations.llamaindex_ import GuardedLlamaIndexAgent

        class FakeAgent:
            def __init__(self, tools: list[Any], **kwargs: Any) -> None:
                self.tools = tools
                self.kwargs = kwargs

            def chat(self, message: str, **kwargs: Any) -> str:
                return self.tools[0].call(record_id="r1").content

        guarded = GuardedLlamaIndexAgent.from_tools(
            [
                FunctionTool.from_defaults(
                    fn=lambda record_id: "deleted", name="delete_record", description="d"
                )
            ],
            policy=PrebuiltRules.read_only(),
            agent_cls=FakeAgent,
        )

        assert "[BLOCKED]" in guarded.chat("delete r1")
        assert len(guarded.blocked) == 1
