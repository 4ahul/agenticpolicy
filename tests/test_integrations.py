"""Guarding logic that every framework adapter shares.

These tests exercise ToolGuard directly, which is what the LangChain,
LlamaIndex and LangGraph adapters delegate to. That keeps the interesting
behaviour covered without requiring any of the three frameworks to be
installed in CI; framework-specific tests below skip cleanly when absent.
"""

from __future__ import annotations

import pytest

from agenticpolicy import Policy, PolicyEngine, PrebuiltRules
from agenticpolicy.core.types import ActionType
from agenticpolicy.exceptions import ApprovalRequired, PolicyViolation
from agenticpolicy.integrations.base import ToolGuard, infer_action, infer_resource


class TestInference:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("salesforce_read_lead", ActionType.READ),
            ("get_ticket", ActionType.READ),
            ("search_docs", ActionType.READ),
            ("update_status", ActionType.WRITE),
            ("create_pr", ActionType.WRITE),
            ("delete_record", ActionType.DELETE),
            ("drop_table", ActionType.DELETE),
            ("deploy_service", ActionType.EXECUTE),
            ("run_query", ActionType.EXECUTE),
        ],
    )
    def test_action_inference(self, name: str, expected: ActionType) -> None:
        assert infer_action(name) is expected

    def test_unknown_names_fail_closed(self) -> None:
        """An unrecognised tool is treated as EXECUTE — the most restricted
        action — rather than as a harmless read."""
        assert infer_action("frobnicate") is ActionType.EXECUTE

    def test_explicit_action_wins(self) -> None:
        assert infer_action("get_ticket", explicit="delete") is ActionType.DELETE

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("salesforce_read_lead", "salesforce:lead"),
            ("github_pr", "github:pr"),
            ("delete_crm_customer", "crm:customer"),
            ("search", "default:search"),
        ],
    )
    def test_resource_inference(self, name: str, expected: str) -> None:
        assert infer_resource(name) == expected

    def test_explicit_resource_wins(self) -> None:
        assert infer_resource("anything", explicit="db:users") == "db:users"


class TestToolGuard:
    def test_allowed_call_runs(self) -> None:
        guard = ToolGuard(PrebuiltRules.read_only())

        def get_ticket(ticket_id: str) -> str:
            return f"ticket {ticket_id}"

        assert guard.wrap(get_ticket)(ticket_id="t1") == "ticket t1"

    def test_blocked_call_never_executes(self) -> None:
        """The point of a guardrail: the side effect must not happen."""
        executed: list[str] = []
        guard = ToolGuard(PrebuiltRules.read_only())

        def delete_record(record_id: str) -> str:
            executed.append(record_id)
            return "deleted"

        result = guard.wrap(delete_record)(record_id="r1")
        assert executed == []
        assert result.startswith("[BLOCKED]")

    def test_positional_args_are_bound_to_names(self) -> None:
        """Conditions written against parameter names must work whether the
        tool is called positionally or by keyword."""
        policy = Policy("x").allow("write", ["crm:ticket"], conditions={"status": ["open"]})
        guard = ToolGuard(
            policy, resource_map={"update": "crm:ticket"}, action_map={"update": "write"}
        )

        def update(ticket_id: str, status: str) -> str:
            return "ok"

        wrapped = guard.wrap(update, name="update")
        assert wrapped("t1", "open") == "ok"
        assert wrapped("t1", "closed").startswith("[BLOCKED]")

    def test_raise_mode(self) -> None:
        guard = ToolGuard(PrebuiltRules.read_only(), on_violation="raise")

        def delete_record(record_id: str) -> str:
            return "deleted"

        with pytest.raises(PolicyViolation) as exc:
            guard.wrap(delete_record)(record_id="r1")
        assert exc.value.tool_call.action is ActionType.DELETE

    def test_approval_raises_its_own_exception(self) -> None:
        guard = ToolGuard(PrebuiltRules.human_in_the_loop(), on_violation="raise")

        def update_record(record_id: str) -> str:
            return "ok"

        with pytest.raises(ApprovalRequired):
            guard.wrap(update_record)(record_id="r1")

    def test_approval_message_is_distinct_from_block(self) -> None:
        guard = ToolGuard(PrebuiltRules.human_in_the_loop())

        def update_record(record_id: str) -> str:
            return "ok"

        assert guard.wrap(update_record)(record_id="r1").startswith("[NEEDS APPROVAL]")

    def test_output_scanning_blocks_leak(self) -> None:
        guard = ToolGuard(PrebuiltRules.data_analyst())

        def read_users() -> str:
            return "name=Ana ssn=123-45-6789"

        assert guard.wrap(read_users, resource="postgres:users")().startswith("[BLOCKED]")

    def test_baseline_context_satisfies_require_context(self) -> None:
        guard = ToolGuard(
            PrebuiltRules.least_privilege_support_bot(),
            context={"user_id": "u1", "ticket_id": "t1"},
        )

        def read_crm_ticket(ticket_id: str) -> str:
            return "open"

        assert guard.wrap(read_crm_ticket)(ticket_id="t1") == "open"

    def test_missing_context_blocks_even_allowed_actions(self) -> None:
        guard = ToolGuard(PrebuiltRules.least_privilege_support_bot())

        def read_crm_ticket(ticket_id: str) -> str:
            return "open"

        assert "context" in guard.wrap(read_crm_ticket)(ticket_id="t1").lower()

    async def test_async_tools_are_supported(self) -> None:
        guard = ToolGuard(PrebuiltRules.read_only())

        async def get_ticket(ticket_id: str) -> str:
            return f"ticket {ticket_id}"

        wrapped = guard.wrap(get_ticket)
        assert await wrapped(ticket_id="t1") == "ticket t1"

    async def test_async_blocked_call_never_executes(self) -> None:
        executed: list[str] = []
        guard = ToolGuard(PrebuiltRules.read_only())

        async def delete_record(record_id: str) -> str:
            executed.append(record_id)
            return "deleted"

        result = await guard.wrap(delete_record)(record_id="r1")
        assert executed == [] and result.startswith("[BLOCKED]")

    def test_wrap_preserves_metadata(self) -> None:
        guard = ToolGuard(PrebuiltRules.read_only())

        def get_ticket(ticket_id: str) -> str:
            """Fetch a ticket."""
            return "x"

        wrapped = guard.wrap(get_ticket)
        assert wrapped.__name__ == "get_ticket"
        assert wrapped.__doc__ == "Fetch a ticket."

    def test_wrap_all(self) -> None:
        guard = ToolGuard(PrebuiltRules.read_only())
        tools = guard.wrap_all({"get_x": lambda: "x", "delete_x": lambda: "gone"})
        assert tools["get_x"]() == "x"
        assert tools["delete_x"]().startswith("[BLOCKED]")

    def test_budget_charged_only_on_execution(self) -> None:
        from tests.conftest import FakeClock

        clock = FakeClock()
        policy = Policy("x").allow("execute", ["ci:*"])
        policy.rate_limit("execute", ["ci:*"], per_hour=1)
        guard = ToolGuard(engine=PolicyEngine(policy, clock=clock))

        def deploy_ci() -> str:
            return "deployed"

        wrapped = guard.wrap(deploy_ci, resource="ci:deploy", action="execute")
        assert wrapped() == "deployed"
        assert wrapped().startswith("[BLOCKED]")

    def test_output_blocked_calls_appear_in_report(self) -> None:
        """Data stopped on the way back is as much a block as a call stopped on
        the way out, and must show up in the run report."""
        guard = ToolGuard(PrebuiltRules.data_analyst(max_output_kb=1))

        def read_postgres_users() -> str:
            return "x" * 5000

        guard.wrap(read_postgres_users, resource="postgres:users")()
        assert len(guard.blocked) == 1
        assert "too large" in guard.report()

    def test_report_lists_blocked_calls(self) -> None:
        guard = ToolGuard(PrebuiltRules.read_only())
        guard.wrap(lambda: "x", name="delete_thing")()
        report = guard.report()
        assert "1 blocked" in report and "delete_thing" in report

    def test_check_does_not_execute(self) -> None:
        guard = ToolGuard(PrebuiltRules.read_only())
        decision = guard.check("delete_record", {"id": "1"})
        assert not decision.allowed

    def test_shared_engine_shares_audit_trail(self, store) -> None:
        engine = PolicyEngine(PrebuiltRules.read_only(), store=store)
        first = ToolGuard(engine=engine)
        second = ToolGuard(engine=engine)
        first.wrap(lambda: "x", name="get_a")()
        second.wrap(lambda: "x", name="delete_b")()
        assert len(store) == 2

    def test_invalid_on_violation_rejected(self) -> None:
        with pytest.raises(ValueError, match="on_violation"):
            ToolGuard(PrebuiltRules.read_only(), on_violation="explode")

    def test_guard_needs_policy_or_engine(self) -> None:
        with pytest.raises(ValueError, match="policy or an engine"):
            ToolGuard()


class TestOptionalDependencies:
    def test_core_imports_without_frameworks(self) -> None:
        """`import agenticpolicy` must not pull in LangChain."""
        import subprocess
        import sys

        code = (
            "import sys, agenticpolicy; "
            "assert 'langchain' not in sys.modules; "
            "assert 'llama_index' not in sys.modules; "
            "print('ok')"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_missing_framework_gives_actionable_error(self) -> None:
        try:
            import langchain_core  # noqa: F401

            pytest.skip("LangChain is installed; nothing to assert about its absence")
        except ImportError:
            pass

        from agenticpolicy.exceptions import IntegrationNotInstalled
        from agenticpolicy.integrations.langchain_ import _require_langchain

        with pytest.raises(IntegrationNotInstalled, match="pip install"):
            _require_langchain()


# Gate only the LangChain-specific class. A module-level importorskip would
# skip this whole file when LangChain is absent — including the ToolGuard tests
# above, which are exactly the ones that must run everywhere.
requires_langchain = pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("langchain_core") is None,
    reason="LangChain not installed",
)


@requires_langchain
class TestLangChain:
    def test_guarded_tool_keeps_name_and_description(self) -> None:
        from langchain_core.tools import StructuredTool

        from agenticpolicy.integrations.langchain_ import guard_tool

        tool = StructuredTool.from_function(
            func=lambda record_id: "deleted",
            name="delete_record",
            description="Delete a record",
        )
        guarded = guard_tool(tool, ToolGuard(PrebuiltRules.read_only()))
        assert guarded.name == "delete_record"
        assert guarded.description == "Delete a record"

    def test_guarded_tool_blocks(self) -> None:
        from langchain_core.tools import StructuredTool

        from agenticpolicy.integrations.langchain_ import guard_tool

        executed: list[str] = []
        tool = StructuredTool.from_function(
            func=lambda record_id: executed.append(record_id) or "deleted",
            name="delete_record",
            description="Delete a record",
        )
        guarded = guard_tool(tool, ToolGuard(PrebuiltRules.read_only()))
        assert guarded.invoke({"record_id": "r1"}).startswith("[BLOCKED]")
        assert executed == []
