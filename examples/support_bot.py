"""Support bot: least-privilege access to tickets and docs.

Runs with no API keys and no LangChain — the tools are plain functions and the
"agent" is a scripted list of calls, so you can see exactly which ones the
policy stops and why.

    python examples/support_bot.py
"""

from __future__ import annotations

from agenticpolicy import EventStore, PolicyEngine, PrebuiltRules
from agenticpolicy.integrations.base import ToolGuard

# --------------------------------------------------------------------- tools
# In a real deployment these would call Salesforce, Zendesk, your docs search.


def read_crm_ticket(ticket_id: str) -> str:
    return f"Ticket {ticket_id}: customer reports a failed payment. Status: open."


def search_kb_articles(query: str) -> str:
    return f"3 knowledge-base articles matched {query!r}."


def write_crm_ticket(ticket_id: str, field: str, value: str) -> str:
    return f"Set {field}={value} on ticket {ticket_id}."


def read_crm_customer(customer_id: str) -> str:
    # Deliberately leaky: the policy's exfiltration scan should catch this.
    return f"Customer {customer_id}: Ana Reyes, ana@example.com, SSN 123-45-6789."


def delete_crm_ticket(ticket_id: str) -> str:
    return f"Deleted ticket {ticket_id}."


def write_billing_refund(customer_id: str, amount: str) -> str:
    return f"Refunded {amount} to {customer_id}."


def main() -> None:
    policy = PrebuiltRules.least_privilege_support_bot()

    print(policy.explain())
    print()

    store = EventStore(":memory:")
    guard = ToolGuard(
        engine=PolicyEngine(policy, store=store),
        # Every call carries who and what it is for — the policy requires both.
        context={"user_id": "agent_42", "ticket_id": "T-8891"},
        # Explicit mapping beats name inference in production: renaming a tool
        # should not silently change which rules apply to it.
        resource_map={
            "read_crm_ticket": "crm:ticket",
            "write_crm_ticket": "crm:ticket",
            "read_crm_customer": "crm:customer",
            "delete_crm_ticket": "crm:ticket",
            "write_billing_refund": "billing:refund",
            "search_kb_articles": "kb:article",
        },
    )

    tools = guard.wrap_all(
        {
            "read_crm_ticket": read_crm_ticket,
            "search_kb_articles": search_kb_articles,
            "write_crm_ticket": write_crm_ticket,
            "read_crm_customer": read_crm_customer,
            "delete_crm_ticket": delete_crm_ticket,
            "write_billing_refund": write_billing_refund,
        }
    )

    # What a support agent might attempt while working one ticket.
    plan = [
        ("read_crm_ticket", {"ticket_id": "T-8891"}, "look up the ticket"),
        ("search_kb_articles", {"query": "failed payment"}, "research the issue"),
        (
            "write_crm_ticket",
            {"ticket_id": "T-8891", "field": "status", "value": "pending"},
            "update the status",
        ),
        (
            "write_crm_ticket",
            {"ticket_id": "T-8891", "field": "owner", "value": "agent_9"},
            "reassign the ticket — only status writes are permitted",
        ),
        ("read_crm_customer", {"customer_id": "C-1"}, "pull the customer record — leaks a SSN"),
        (
            "write_billing_refund",
            {"customer_id": "C-1", "amount": "49.00"},
            "issue a refund — billing is off-limits",
        ),
        ("delete_crm_ticket", {"ticket_id": "T-8891"}, "delete the ticket — never permitted"),
    ]

    for name, kwargs, intent in plan:
        result = tools[name](**kwargs)
        verdict = "BLOCKED" if str(result).startswith("[") else "ran"
        print(f"[{verdict:>7}] {name}  ({intent})")
        print(f"          {result}")

    print()
    print(guard.report())
    print()
    summary = store.summary()
    print(
        f"Audit: {summary['total_calls']} calls, {summary['blocked']} blocked "
        f"({summary['block_rate']:.0%} block rate)"
    )


if __name__ == "__main__":
    main()
