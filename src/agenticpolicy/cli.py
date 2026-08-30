"""Command-line interface: ``agenticpolicy <command>``.

Commands:

``audit``
    Show recent decisions from an audit database.

``summary``
    Aggregate block rate and top blocked resources.

``explain``
    Print a saved policy JSON file in readable form.

``test``
    Check a single hypothetical tool call against a saved policy — useful in CI
    to assert a policy still denies what it should.

``catalog``
    List the built-in policies.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

from agenticpolicy import __version__
from agenticpolicy.audit.store import EventStore
from agenticpolicy.core.engine import PolicyEngine
from agenticpolicy.core.policy import Policy
from agenticpolicy.core.rules import PrebuiltRules
from agenticpolicy.core.types import ToolCall


def _cmd_audit(args: argparse.Namespace) -> int:
    store = EventStore(args.db)
    since = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    events = store.query_events(
        args.agent,
        allowed=False if args.blocked_only else None,
        since=since,
        limit=args.limit,
    )
    if not events:
        print(f"No events in the last {args.hours}h.")
        return 0

    print(f"{'TIME':<20} {'AGENT':<18} {'ACTION':<8} {'RESOURCE':<26} {'VERDICT':<14} REASON")
    print("-" * 118)
    for e in events:
        verdict = "ALLOW" if e.allowed else e.effect.upper()
        print(
            f"{e.timestamp.strftime('%Y-%m-%d %H:%M:%S'):<20} {e.agent_id[:17]:<18} "
            f"{e.action:<8} {e.resource[:25]:<26} {verdict:<14} {e.reason[:44]}"
        )
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    store = EventStore(args.db)
    data = store.summary(args.agent, hours=args.hours)
    if args.json:
        print(json.dumps(data, indent=2))
        return 0

    print(f"Window:      last {data['window_hours']}h")
    print(f"Total calls: {data['total_calls']}")
    print(f"Allowed:     {data['allowed']}")
    print(f"Blocked:     {data['blocked']} ({data['block_rate']:.1%})")
    print(f"Avg risk:    {data['avg_risk_score']:.2f}")
    if data["top_blocked"]:
        print("\nMost blocked:")
        for row in data["top_blocked"]:
            print(f"  {row['count']:>4}x  {row['action']:<8} {row['resource']}")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    policy = Policy.load(args.policy)
    print(policy.explain())
    return 0


def _cmd_test(args: argparse.Namespace) -> int:
    policy = Policy.load(args.policy)
    engine = PolicyEngine(policy)
    context = json.loads(args.context) if args.context else {}
    call = ToolCall(
        agent_id=policy.agent_id,
        tool_name=args.tool or f"{args.action}_{args.resource}",
        resource=args.resource,
        action=args.action,
        args=json.loads(args.args) if args.args else {},
        context=context,
    )
    decision = engine.dry_run(call)
    print(decision)
    if decision.rule_applied:
        print(f"  rule: {decision.rule_applied}")
    for finding in decision.findings:
        print(f"  finding: {finding}")

    if args.expect:
        actual = decision.effect.value
        if actual != args.expect:
            print(f"FAIL: expected {args.expect}, got {actual}", file=sys.stderr)
            return 1
        print(f"OK: expected {args.expect}")
    return 0 if decision.allowed or not args.expect else 0


def _cmd_catalog(args: argparse.Namespace) -> int:
    for name, doc in PrebuiltRules.catalog().items():
        print(f"{name}\n    {doc}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agenticpolicy", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"agenticpolicy {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="show recent policy decisions")
    audit.add_argument("--db", default="agenticpolicy.db")
    audit.add_argument("--agent", default=None, help="filter to one agent id")
    audit.add_argument("--hours", type=int, default=24)
    audit.add_argument("--limit", type=int, default=50)
    audit.add_argument("--blocked-only", action="store_true")
    audit.set_defaults(func=_cmd_audit)

    summary = sub.add_parser("summary", help="aggregate stats from the audit log")
    summary.add_argument("--db", default="agenticpolicy.db")
    summary.add_argument("--agent", default=None)
    summary.add_argument("--hours", type=int, default=24)
    summary.add_argument("--json", action="store_true")
    summary.set_defaults(func=_cmd_summary)

    explain = sub.add_parser("explain", help="print a saved policy in readable form")
    explain.add_argument("policy", help="path to a policy JSON file")
    explain.set_defaults(func=_cmd_explain)

    test = sub.add_parser("test", help="check one hypothetical call against a policy")
    test.add_argument("policy", help="path to a policy JSON file")
    test.add_argument("action", help="read | write | delete | execute")
    test.add_argument("resource", help='e.g. "salesforce:lead"')
    test.add_argument("--tool", default=None)
    test.add_argument("--args", default=None, help="JSON object of tool arguments")
    test.add_argument("--context", default=None, help="JSON object of call context")
    test.add_argument(
        "--expect",
        choices=["allow", "deny", "needs_approval"],
        default=None,
        help="assert the outcome; exits 1 on mismatch (for CI)",
    )
    test.set_defaults(func=_cmd_test)

    catalog = sub.add_parser("catalog", help="list built-in policies")
    catalog.set_defaults(func=_cmd_catalog)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
