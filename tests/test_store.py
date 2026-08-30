"""The SQLite audit store."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from agenticpolicy import EventStore, PolicyEngine
from agenticpolicy.core.types import PolicyDecision, PolicyEvent
from tests.conftest import make_call


def _event(allowed: bool = True, agent: str = "a1", resource: str = "crm:ticket") -> PolicyEvent:
    call = make_call("read", resource, agent_id=agent)
    decision = (
        PolicyDecision.allow("ok", rule_applied="allow_0")
        if allowed
        else PolicyDecision.deny("nope", rule_applied="deny_0", risk_score=0.9)
    )
    return PolicyEvent.from_decision(call, decision)


class TestWriteAndRead:
    def test_log_and_count(self, store: EventStore) -> None:
        store.log_event(_event())
        store.log_event(_event(allowed=False))
        assert len(store) == 2

    def test_round_trip_preserves_fields(self, store: EventStore) -> None:
        original = _event(allowed=False)
        store.log_event(original)
        restored = store.query_events()[0]
        assert restored.id == original.id
        assert restored.allowed is False
        assert restored.risk_score == 0.9
        assert restored.context == original.context
        assert restored.rule_applied == "deny_0"

    def test_newest_first(self, store: EventStore) -> None:
        old = _event()
        old.timestamp = datetime.now(timezone.utc) - timedelta(hours=2)
        new = _event()
        store.log_event(old)
        store.log_event(new)
        assert store.query_events()[0].id == new.id

    def test_idempotent_on_same_id(self, store: EventStore) -> None:
        event = _event()
        store.log_event(event)
        store.log_event(event)
        assert len(store) == 1


class TestFiltering:
    def test_by_agent(self, store: EventStore) -> None:
        store.log_event(_event(agent="a1"))
        store.log_event(_event(agent="a2"))
        assert len(store.query_events("a1")) == 1

    def test_by_allowed(self, store: EventStore) -> None:
        store.log_event(_event(allowed=True))
        store.log_event(_event(allowed=False))
        store.log_event(_event(allowed=False))
        assert len(store.blocked_events()) == 2

    def test_by_resource(self, store: EventStore) -> None:
        store.log_event(_event(resource="crm:ticket"))
        store.log_event(_event(resource="db:users"))
        assert len(store.query_events(resource="db:users")) == 1

    def test_by_since(self, store: EventStore) -> None:
        old = _event()
        old.timestamp = datetime.now(timezone.utc) - timedelta(days=2)
        store.log_event(old)
        store.log_event(_event())
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        assert len(store.query_events(since=cutoff)) == 1

    def test_limit(self, store: EventStore) -> None:
        for _ in range(10):
            store.log_event(_event())
        assert len(store.query_events(limit=3)) == 3

    def test_agents_listing(self, store: EventStore) -> None:
        store.log_event(_event(agent="b"))
        store.log_event(_event(agent="a"))
        assert store.agents() == ["a", "b"]


class TestSummary:
    def test_counts_and_block_rate(self, store: EventStore) -> None:
        for _ in range(3):
            store.log_event(_event(allowed=True))
        store.log_event(_event(allowed=False))
        summary = store.summary()
        assert summary["total_calls"] == 4
        assert summary["blocked"] == 1
        assert summary["block_rate"] == 0.25

    def test_empty_store_does_not_divide_by_zero(self, store: EventStore) -> None:
        assert store.summary()["block_rate"] == 0.0

    def test_top_blocked_is_ranked(self, store: EventStore) -> None:
        for _ in range(3):
            store.log_event(_event(allowed=False, resource="db:secrets"))
        store.log_event(_event(allowed=False, resource="crm:ticket"))
        top = store.summary()["top_blocked"]
        assert top[0]["resource"] == "db:secrets" and top[0]["count"] == 3


class TestMaintenance:
    def test_export_jsonl(self, store: EventStore, tmp_path) -> None:
        store.log_event(_event())
        store.log_event(_event(allowed=False))
        path = tmp_path / "audit.jsonl"
        assert store.export_jsonl(path) == 2
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["agent_id"] == "a1"

    def test_purge_old_events(self, store: EventStore) -> None:
        old = _event()
        old.timestamp = datetime.now(timezone.utc) - timedelta(days=90)
        store.log_event(old)
        store.log_event(_event())
        removed = store.purge(before=datetime.now(timezone.utc) - timedelta(days=30))
        assert removed == 1 and len(store) == 1

    def test_file_backed_store_persists(self, tmp_path) -> None:
        path = tmp_path / "nested" / "audit.db"
        with EventStore(path) as first:
            first.log_event(_event())
        with EventStore(path) as second:
            assert len(second) == 1

    def test_engine_writes_through(self, support_policy, store: EventStore) -> None:
        engine = PolicyEngine(support_policy, store=store)
        engine.evaluate_sync(make_call("delete", "salesforce:lead"))
        event = store.query_events()[0]
        assert event.allowed is False and event.action == "delete"
