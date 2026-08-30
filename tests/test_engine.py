"""Policy evaluation: the ordering of checks, budgets, and exfiltration."""

from __future__ import annotations

import pytest

from agenticpolicy import Policy, PolicyEngine
from agenticpolicy.core.types import Effect
from tests.conftest import FakeClock, make_call


class TestBasicEvaluation:
    async def test_allow_read(self, engine: PolicyEngine) -> None:
        decision = await engine.evaluate(make_call("read", "salesforce:lead"))
        assert decision.allowed
        assert decision.rule_applied == "allow_0"

    async def test_deny_delete(self, engine: PolicyEngine) -> None:
        decision = await engine.evaluate(make_call("delete", "salesforce:lead"))
        assert not decision.allowed
        assert "Denied by rule" in decision.reason

    async def test_missing_context_blocks(self, engine: PolicyEngine) -> None:
        decision = await engine.evaluate(make_call("read", context={"user_id": "u1"}))
        assert not decision.allowed
        assert "ticket_id" in decision.reason

    async def test_default_deny(self, engine: PolicyEngine) -> None:
        """A resource no allow rule mentions is refused."""
        decision = await engine.evaluate(make_call("read", "stripe:charge"))
        assert not decision.allowed
        assert "No allow rule" in decision.reason

    async def test_write_needs_approval(self, engine: PolicyEngine) -> None:
        decision = await engine.evaluate(make_call("write", "salesforce:lead"))
        assert not decision.allowed
        assert decision.requires_approval
        assert decision.effect is Effect.NEEDS_APPROVAL

    def test_decision_is_truthy_when_allowed(self, engine: PolicyEngine) -> None:
        assert bool(engine.dry_run(make_call("read"))) is True
        assert bool(engine.dry_run(make_call("delete"))) is False


class TestOrdering:
    def test_deny_beats_allow(self) -> None:
        policy = Policy("x").allow("read", ["db:*"]).deny("read", ["db:secrets"])
        engine = PolicyEngine(policy)
        assert engine.dry_run(make_call("read", "db:public", context={})).allowed
        assert not engine.dry_run(make_call("read", "db:secrets", context={})).allowed

    def test_context_checked_before_rules(self) -> None:
        """Missing context is reported even when a deny rule would also fire,
        so the operator sees the root cause rather than a downstream symptom."""
        policy = Policy("x").deny("delete", ["*"]).require_context(["user_id"])
        decision = PolicyEngine(policy).dry_run(make_call("delete", context={}))
        assert "Missing required context" in decision.reason

    def test_wildcard_action_matches_everything(self) -> None:
        policy = Policy("x").allow("*", ["db:*"])
        engine = PolicyEngine(policy)
        for action in ("read", "write", "delete", "execute"):
            assert engine.dry_run(make_call(action, "db:x", context={})).allowed

    def test_near_miss_is_explained(self) -> None:
        """A denial caused by a failed condition names the condition."""
        policy = Policy("x").allow("write", ["crm:ticket"], conditions={"status": ["open"]})
        decision = PolicyEngine(policy).dry_run(
            make_call("write", "crm:ticket", args={"status": "closed"}, context={})
        )
        assert not decision.allowed
        assert "status" in decision.reason


class TestConditionsInEvaluation:
    def test_condition_from_args(self) -> None:
        policy = Policy("x").allow(
            "write", ["crm:ticket"], conditions={"status": ["open", "pending"]}
        )
        engine = PolicyEngine(policy)
        assert engine.dry_run(
            make_call("write", "crm:ticket", args={"status": "open"}, context={})
        ).allowed
        assert not engine.dry_run(
            make_call("write", "crm:ticket", args={"status": "closed"}, context={})
        ).allowed

    def test_context_overrides_args(self) -> None:
        """Trusted execution context wins over model-supplied arguments — an
        agent must not be able to talk its way past a condition."""
        policy = Policy("x").allow("write", ["crm:ticket"], conditions={"role": "admin"})
        engine = PolicyEngine(policy)
        call = make_call("write", "crm:ticket", args={"role": "admin"}, context={"role": "guest"})
        assert not engine.dry_run(call).allowed


class TestBudgets:
    def test_rate_limit_allows_then_blocks(self, clock: FakeClock) -> None:
        policy = Policy("x").allow("execute", ["ci:*"])
        policy.rate_limit("execute", ["ci:*"], per_hour=2)
        engine = PolicyEngine(policy, clock=clock)

        for _ in range(2):
            call = make_call("execute", "ci:deploy", context={})
            assert engine.evaluate_sync(call).allowed
            engine.commit(call)

        assert not engine.evaluate_sync(make_call("execute", "ci:deploy", context={})).allowed

    def test_window_slides(self, clock: FakeClock) -> None:
        """Budget is per rolling hour, not per process lifetime."""
        policy = Policy("x").allow("execute", ["ci:*"])
        policy.rate_limit("execute", ["ci:*"], per_hour=1)
        engine = PolicyEngine(policy, clock=clock)

        call = make_call("execute", "ci:deploy", context={})
        assert engine.evaluate_sync(call).allowed
        engine.commit(call)
        assert not engine.evaluate_sync(make_call("execute", "ci:deploy", context={})).allowed

        clock.advance(3601)
        assert engine.evaluate_sync(make_call("execute", "ci:deploy", context={})).allowed

    def test_uncommitted_calls_do_not_consume_budget(self, clock: FakeClock) -> None:
        """Evaluating without executing must not spend the user's quota."""
        policy = Policy("x").allow("execute", ["ci:*"])
        policy.rate_limit("execute", ["ci:*"], per_hour=1)
        engine = PolicyEngine(policy, clock=clock)

        engine.evaluate_sync(make_call("execute", "ci:deploy", context={}))  # never committed
        assert engine.evaluate_sync(make_call("execute", "ci:deploy", context={})).allowed

    def test_per_task_budget_resets_per_task(self, clock: FakeClock) -> None:
        policy = Policy("x").allow("execute", ["ci:*"])
        policy.rate_limit("execute", ["ci:*"], per_task=1)
        engine = PolicyEngine(policy, clock=clock)

        first = make_call("execute", "ci:deploy", context={"task_id": "t1"})
        assert engine.evaluate_sync(first).allowed
        engine.commit(first)
        assert not engine.evaluate_sync(
            make_call("execute", "ci:deploy", context={"task_id": "t1"})
        ).allowed
        assert engine.evaluate_sync(
            make_call("execute", "ci:deploy", context={"task_id": "t2"})
        ).allowed

    def test_approval_gate_with_budget_still_gates(self, clock: FakeClock) -> None:
        """require_approval + budget means both apply, not either."""
        policy = Policy("x").allow("execute", ["ci:*"])
        policy.require_approval("execute", ["ci:deploy"], budget={"per_day": 5})
        decision = PolicyEngine(policy, clock=clock).dry_run(
            make_call("execute", "ci:deploy", context={})
        )
        assert decision.requires_approval

    def test_budgets_are_per_agent(self, clock: FakeClock) -> None:
        policy = Policy("shared").allow("execute", ["ci:*"])
        policy.rate_limit("execute", ["ci:*"], per_hour=1)
        engine = PolicyEngine(policy, clock=clock)

        a = make_call("execute", "ci:deploy", agent_id="agent_a", context={})
        engine.evaluate_sync(a)
        engine.commit(a)
        b = make_call("execute", "ci:deploy", agent_id="agent_b", context={})
        assert engine.evaluate_sync(b).allowed


class TestExfiltration:
    def test_output_size_limit(self, analyst_policy: Policy) -> None:
        engine = PolicyEngine(analyst_policy)
        call = make_call("read", "postgres:users", context={})
        assert engine.dry_run(call).allowed

        decision, safe = engine.check_output(call, "x" * 5000)  # ~5KB, limit is 1KB
        assert not decision.allowed
        assert "too large" in decision.reason
        assert safe is None

    def test_output_pattern_blocked(self, analyst_policy: Policy) -> None:
        engine = PolicyEngine(analyst_policy)
        call = make_call("read", "postgres:users", context={})
        decision, safe = engine.check_output(call, "SSN on file: 123-45-6789")
        assert not decision.allowed
        assert decision.findings
        assert safe is None

    def test_findings_do_not_quote_the_secret(self, analyst_policy: Policy) -> None:
        """An audit log that echoes the SSN it caught is its own data leak."""
        engine = PolicyEngine(analyst_policy)
        call = make_call("read", "postgres:users", context={})
        decision, _ = engine.check_output(call, "SSN: 123-45-6789")
        assert all("123-45-6789" not in f for f in decision.findings)

    def test_redact_mode_lets_the_call_through(self) -> None:
        policy = Policy("x").allow("read", ["db:*"])
        policy.prevent_exfiltration(block_patterns=[r"\b\d{3}-\d{2}-\d{4}\b"], redact=True)
        engine = PolicyEngine(policy)
        call = make_call("read", "db:users", context={})
        decision, safe = engine.check_output(call, "SSN 123-45-6789 belongs to Ana")
        assert decision.allowed
        assert "123-45-6789" not in safe
        assert "Ana" in safe

    def test_outbound_args_are_scanned(self) -> None:
        """The exfiltration vector is usually what the agent *sends*, which a
        post-call-only design never sees."""
        policy = Policy("x").allow("write", ["http:webhook"])
        policy.prevent_exfiltration(block_patterns=[r"\b\d{3}-\d{2}-\d{4}\b"])
        engine = PolicyEngine(policy)
        decision = engine.dry_run(
            make_call("write", "http:webhook", args={"body": "ssn=123-45-6789"}, context={})
        )
        assert not decision.allowed
        assert "arguments" in decision.reason

    def test_clean_output_passes_through_unchanged(self, analyst_policy: Policy) -> None:
        engine = PolicyEngine(analyst_policy)
        call = make_call("read", "postgres:users", context={})
        decision, safe = engine.check_output(call, "42 rows")
        assert decision.allowed and safe == "42 rows"

    def test_no_exfil_policy_is_a_no_op(self) -> None:
        engine = PolicyEngine(Policy("x").allow("read", ["*"]))
        call = make_call("read", "db:x", context={})
        decision, safe = engine.check_output(call, "123-45-6789")
        assert decision.allowed and safe == "123-45-6789"

    def test_malformed_pattern_does_not_crash(self) -> None:
        policy = Policy("x").allow("read", ["db:*"])
        policy.prevent_exfiltration(block_patterns=["[unclosed"])
        engine = PolicyEngine(policy)
        call = make_call("read", "db:x", context={})
        decision, _ = engine.check_output(call, "anything")
        assert "invalid pattern" in " ".join(decision.findings)

    def test_non_string_output_is_scanned(self, analyst_policy: Policy) -> None:
        """Dicts and lists get flattened before scanning, so structured tool
        results are covered too."""
        engine = PolicyEngine(analyst_policy)
        call = make_call("read", "postgres:users", context={})
        decision, _ = engine.check_output(call, {"rows": [{"ssn": "123-45-6789"}]})
        assert not decision.allowed


class TestAuditIntegration:
    def test_decisions_are_logged(self, support_policy: Policy, store) -> None:
        engine = PolicyEngine(support_policy, store=store)
        engine.evaluate_sync(make_call("read", "salesforce:lead"))
        engine.evaluate_sync(make_call("delete", "salesforce:lead"))
        assert len(store) == 2
        assert len(store.blocked_events()) == 1

    def test_dry_run_does_not_log(self, support_policy: Policy, store) -> None:
        PolicyEngine(support_policy, store=store).dry_run(make_call("read"))
        assert len(store) == 0

    def test_broken_store_does_not_break_the_agent(self, support_policy: Policy) -> None:
        class ExplodingStore:
            def log_event(self, event):
                raise RuntimeError("disk full")

        engine = PolicyEngine(support_policy, store=ExplodingStore())
        assert engine.evaluate_sync(make_call("read")).allowed


class TestPerformance:
    def test_evaluation_is_fast(self, engine: PolicyEngine) -> None:
        """The design target is <5ms per call; patterns are cached, so a
        thousand evaluations should finish comfortably inside a second."""
        import time

        call = make_call("read", "salesforce:lead")
        start = time.perf_counter()
        for _ in range(1000):
            engine.dry_run(call)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"1000 evaluations took {elapsed:.3f}s"


@pytest.mark.parametrize("action", ["read", "write", "delete", "execute"])
def test_every_action_is_evaluable(action: str) -> None:
    engine = PolicyEngine(Policy("x").allow("*", ["*"]))
    assert engine.dry_run(make_call(action, "any:thing", context={})).allowed
