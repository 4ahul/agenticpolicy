"""Resource pattern matching and condition evaluation.

Resource patterns look like ``provider:type`` — ``salesforce:lead``,
``github:repo``, ``postgres:users``. Two wildcards are supported:

``*``
    Matches any characters *within a single segment*. ``crm:*`` matches
    ``crm:ticket`` but **not** ``salesforce:lead``, and ``*:lead`` matches
    ``salesforce:lead`` but not ``salesforce:contact``.

``**``
    Matches any characters, including ``:``. ``github:**`` matches
    ``github:repo`` and ``github:repo:main``.

A pattern that is exactly ``*`` is shorthand for ``**`` — it matches
everything, which is what people mean when they write ``deny("delete",
resources=["*"])``.

Matching is case-insensitive: ``Salesforce:Lead`` matches ``salesforce:*``.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

__all__ = ["matches_resource", "compile_pattern", "check_conditions", "resolve_field"]

# Sentinels that survive re.escape unchanged, used to swap wildcards back in.
_DOUBLE = "\x00DOUBLESTAR\x00"
_SINGLE = "\x00SINGLESTAR\x00"


@lru_cache(maxsize=1024)
def compile_pattern(pattern: str) -> re.Pattern[str]:
    """Compile a resource pattern into an anchored, case-insensitive regex.

    Results are cached, so hot-path evaluation does no regex compilation.
    """
    if pattern.strip() == "*":
        return re.compile(r".*", re.IGNORECASE)

    # Swap wildcards for sentinels, escape everything else, swap back.
    staged = pattern.replace("**", _DOUBLE).replace("*", _SINGLE)
    escaped = re.escape(staged)
    escaped = escaped.replace(re.escape(_DOUBLE), ".*")
    escaped = escaped.replace(re.escape(_SINGLE), "[^:]*")
    return re.compile(f"^{escaped}$", re.IGNORECASE)


def matches_resource(pattern: str, resource: str) -> bool:
    """True if ``resource`` matches ``pattern``.

    Unlike a naive ``pattern.replace("*", ".*")``, regex metacharacters in the
    pattern are escaped, so a resource named ``crm.ticket`` does not match the
    pattern ``crm:ticket``, and ``*`` does not silently span segments.
    """
    return compile_pattern(pattern).match(resource) is not None


def resolve_field(field: str, *sources: dict[str, Any]) -> tuple[bool, Any]:
    """Look up ``field`` across sources in order, supporting ``a.b`` paths.

    Returns ``(found, value)`` so that a legitimately stored ``None`` is
    distinguishable from a missing field.
    """
    parts = field.split(".")
    for source in sources:
        cursor: Any = source
        ok = True
        for part in parts:
            if isinstance(cursor, dict) and part in cursor:
                cursor = cursor[part]
            else:
                ok = False
                break
        if ok:
            return True, cursor
    return False, None


def _compare(op: str, actual: Any, expected: Any) -> bool:
    """Apply a single comparison operator, returning False on type mismatch."""
    try:
        if op == "eq":
            return bool(actual == expected)
        if op == "ne":
            return bool(actual != expected)
        if op == "in":
            return bool(actual in expected)
        if op == "not_in":
            return bool(actual not in expected)
        if op == "gt":
            return bool(actual > expected)
        if op == "gte":
            return bool(actual >= expected)
        if op == "lt":
            return bool(actual < expected)
        if op == "lte":
            return bool(actual <= expected)
        if op == "contains":
            return bool(expected in actual)
        if op == "matches":
            return re.search(str(expected), str(actual)) is not None
        if op == "exists":
            return bool(expected)
    except TypeError:
        # e.g. comparing a string amount to an int limit — treat as no match
        # rather than crashing the agent mid-run.
        return False
    raise ValueError(
        f"Unknown condition operator {op!r}. Valid operators: eq, ne, in, not_in, "
        "gt, gte, lt, lte, contains, matches, exists"
    )


def check_conditions(
    conditions: dict[str, Any] | None,
    *sources: dict[str, Any],
) -> tuple[bool, str | None]:
    """Evaluate a rule's conditions against the given lookup sources.

    Three value shapes are accepted::

        {"status": "open"}                  # equality
        {"status": ["open", "pending"]}     # membership
        {"amount": {"gte": 1000}}           # explicit operators

    Sources are searched in order — conventionally the tool call's ``context``
    first, then its ``args`` — so a caller can override an argument with
    trusted execution context.

    Returns ``(passed, failure_reason)``.
    """
    if not conditions:
        return True, None

    for field, expected in conditions.items():
        found, actual = resolve_field(field, *sources)

        if isinstance(expected, dict):
            if not found and set(expected) != {"exists"}:
                return False, f"condition field {field!r} is missing"
            for op, operand in expected.items():
                if op == "exists":
                    if bool(operand) != found:
                        return False, f"condition field {field!r} existence check failed"
                    continue
                if not _compare(op, actual, operand):
                    return False, f"condition {field}.{op} failed (actual={actual!r})"
            continue

        if not found:
            return False, f"condition field {field!r} is missing"

        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False, f"condition {field!r}={actual!r} not in {sorted(expected, key=str)}"
        elif actual != expected:
            return False, f"condition {field!r}={actual!r} != {expected!r}"

    return True, None
