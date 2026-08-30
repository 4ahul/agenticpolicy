"""SQLite audit store for policy decisions.

Built on the standard library's ``sqlite3`` rather than SQLAlchemy, so the
audit trail — the part you most want present in production — carries no
third-party dependency and cannot fail to import.

The table is append-only by convention: nothing in this module updates or
deletes a row except :meth:`EventStore.purge`, which is explicit about it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from agenticpolicy.core.types import PolicyEvent

__all__ = ["EventStore"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS policy_events (
    id            TEXT PRIMARY KEY,
    timestamp     TEXT    NOT NULL,
    agent_id      TEXT    NOT NULL,
    tool_name     TEXT    NOT NULL,
    resource      TEXT    NOT NULL,
    action        TEXT    NOT NULL,
    allowed       INTEGER NOT NULL,
    effect        TEXT    NOT NULL,
    reason        TEXT    NOT NULL,
    rule_applied  TEXT,
    risk_score    REAL    NOT NULL DEFAULT 0.0,
    context_json  TEXT    NOT NULL DEFAULT '{}',
    findings_json TEXT    NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_events_agent     ON policy_events(agent_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON policy_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_allowed   ON policy_events(allowed);
CREATE INDEX IF NOT EXISTS idx_events_resource  ON policy_events(resource);
"""


class EventStore:
    """Append-only store of :class:`PolicyEvent` records.

    Args:
        db_path: SQLite file, or ``":memory:"`` for an ephemeral store.

    Safe to share across threads: each thread gets its own connection, which
    is what SQLite requires.
    """

    def __init__(self, db_path: str | Path = "agenticpolicy.db") -> None:
        self.db_path = str(db_path)
        self._local = threading.local()
        self._memory_conn: sqlite3.Connection | None = None
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # An in-memory database lives only as long as its connection, so it
        # must be shared rather than opened per thread.
        if self.db_path == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._memory_conn.row_factory = sqlite3.Row
            return self._memory_conn

        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    # -------------------------------------------------------------- writing

    def log_event(self, event: PolicyEvent) -> str:
        """Persist one decision. Returns the event id."""
        conn = self._connect()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO policy_events
                    (id, timestamp, agent_id, tool_name, resource, action,
                     allowed, effect, reason, rule_applied, risk_score,
                     context_json, findings_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    event.id,
                    event.timestamp.isoformat(),
                    event.agent_id,
                    event.tool_name,
                    event.resource,
                    event.action,
                    int(event.allowed),
                    event.effect,
                    event.reason,
                    event.rule_applied,
                    event.risk_score,
                    json.dumps(event.context, default=str),
                    json.dumps(event.findings, default=str),
                ),
            )
        return event.id

    # -------------------------------------------------------------- reading

    def query_events(
        self,
        agent_id: str | None = None,
        *,
        allowed: bool | None = None,
        resource: str | None = None,
        action: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[PolicyEvent]:
        """Fetch events newest-first, filtered by any combination of criteria."""
        clauses: list[str] = []
        params: list[Any] = []
        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if allowed is not None:
            clauses.append("allowed = ?")
            params.append(int(allowed))
        if resource is not None:
            clauses.append("resource = ?")
            params.append(resource)
        if action is not None:
            clauses.append("action = ?")
            params.append(action)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since.isoformat())

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = (
            self._connect()
            .execute(
                f"SELECT * FROM policy_events {where} ORDER BY timestamp DESC LIMIT ?",
                (*params, limit),
            )
            .fetchall()
        )
        return [self._row_to_event(r) for r in rows]

    def blocked_events(self, agent_id: str | None = None, limit: int = 100) -> list[PolicyEvent]:
        """Just the denials — usually the only rows anyone reads."""
        return self.query_events(agent_id, allowed=False, limit=limit)

    def summary(self, agent_id: str | None = None, hours: int = 24) -> dict[str, Any]:
        """Aggregate stats for a dashboard or a CLI report."""
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        params: list[Any] = [since.isoformat()]
        agent_clause = ""
        if agent_id is not None:
            agent_clause = "AND agent_id = ?"
            params.append(agent_id)

        conn = self._connect()
        totals = conn.execute(
            f"""
            SELECT COUNT(*) AS total,
                   SUM(allowed) AS allowed,
                   AVG(risk_score) AS avg_risk
            FROM policy_events WHERE timestamp >= ? {agent_clause}
            """,
            params,
        ).fetchone()

        total = totals["total"] or 0
        allowed = totals["allowed"] or 0
        top_blocked = conn.execute(
            f"""
            SELECT resource, action, COUNT(*) AS n
            FROM policy_events
            WHERE timestamp >= ? {agent_clause} AND allowed = 0
            GROUP BY resource, action ORDER BY n DESC LIMIT 10
            """,
            params,
        ).fetchall()

        return {
            "window_hours": hours,
            "total_calls": total,
            "allowed": allowed,
            "blocked": total - allowed,
            "block_rate": round((total - allowed) / total, 4) if total else 0.0,
            "avg_risk_score": round(totals["avg_risk"] or 0.0, 4),
            "top_blocked": [
                {"resource": r["resource"], "action": r["action"], "count": r["n"]}
                for r in top_blocked
            ],
        }

    def agents(self) -> list[str]:
        """Every agent id that has logged at least one event."""
        rows = (
            self._connect()
            .execute("SELECT DISTINCT agent_id FROM policy_events ORDER BY agent_id")
            .fetchall()
        )
        return [r["agent_id"] for r in rows]

    def export_jsonl(self, path: str | Path, agent_id: str | None = None) -> int:
        """Write the audit trail to newline-delimited JSON. Returns the row count."""
        events = self.query_events(agent_id, limit=1_000_000)
        with open(path, "w", encoding="utf-8") as fh:
            for event in events:
                fh.write(
                    json.dumps(
                        {
                            "id": event.id,
                            "timestamp": event.timestamp.isoformat(),
                            "agent_id": event.agent_id,
                            "tool_name": event.tool_name,
                            "resource": event.resource,
                            "action": event.action,
                            "allowed": event.allowed,
                            "effect": event.effect,
                            "reason": event.reason,
                            "rule_applied": event.rule_applied,
                            "risk_score": event.risk_score,
                            "context": event.context,
                            "findings": event.findings,
                        },
                        default=str,
                    )
                    + "\n"
                )
        return len(events)

    def purge(self, before: datetime) -> int:
        """Delete events older than ``before``. Returns rows removed.

        The one destructive operation in this module — for retention policies,
        not for hiding a decision after the fact.
        """
        conn = self._connect()
        with conn:
            cur = conn.execute(
                "DELETE FROM policy_events WHERE timestamp < ?", (before.isoformat(),)
            )
        return cur.rowcount

    def close(self) -> None:
        """Close this thread's connection."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
        if self._memory_conn is not None:
            self._memory_conn.close()
            self._memory_conn = None

    def __enter__(self) -> EventStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[PolicyEvent]:
        return iter(self.query_events(limit=1_000_000))

    def __len__(self) -> int:
        return int(self._connect().execute("SELECT COUNT(*) FROM policy_events").fetchone()[0])

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> PolicyEvent:
        return PolicyEvent(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            agent_id=row["agent_id"],
            tool_name=row["tool_name"],
            resource=row["resource"],
            action=row["action"],
            allowed=bool(row["allowed"]),
            effect=row["effect"],
            reason=row["reason"],
            rule_applied=row["rule_applied"],
            risk_score=row["risk_score"],
            context=json.loads(row["context_json"]),
            findings=json.loads(row["findings_json"]),
        )
