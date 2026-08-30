"""Resource pattern matching and condition evaluation.

These are the tests that catch the bug in the original design: a naive
``pattern.replace("*", ".*")`` makes ``crm:*`` match ``crm-anything`` and lets
regex metacharacters in resource names change what a rule covers.
"""

from __future__ import annotations

import pytest

from agenticpolicy.core.matching import check_conditions, matches_resource, resolve_field


class TestResourceMatching:
    @pytest.mark.parametrize(
        "pattern,resource",
        [
            ("salesforce:lead", "salesforce:lead"),
            ("salesforce:*", "salesforce:lead"),
            ("salesforce:*", "salesforce:contact"),
            ("*:lead", "salesforce:lead"),
            ("*", "anything:at:all"),
            ("github:**", "github:repo:main"),
            ("Salesforce:Lead", "salesforce:lead"),  # case-insensitive
        ],
    )
    def test_matches(self, pattern: str, resource: str) -> None:
        assert matches_resource(pattern, resource)

    @pytest.mark.parametrize(
        "pattern,resource",
        [
            ("salesforce:lead", "salesforce:contact"),
            ("crm:*", "salesforce:lead"),
            ("*:lead", "salesforce:contact"),
            ("github:repo", "github:repo:main"),  # single * does not cross ':'
        ],
    )
    def test_does_not_match(self, pattern: str, resource: str) -> None:
        assert not matches_resource(pattern, resource)

    def test_single_star_does_not_span_segments(self) -> None:
        """`crm:*` must not reach into a deeper namespace."""
        assert matches_resource("crm:*", "crm:ticket")
        assert not matches_resource("crm:*", "crm:ticket:comment")
        assert matches_resource("crm:**", "crm:ticket:comment")

    def test_metacharacters_are_escaped(self) -> None:
        """A '.' in a pattern is a literal dot, not 'any character'.

        Without escaping, the pattern 'crm.ticket' would match the unrelated
        resource 'crm:ticket' and silently widen the rule.
        """
        assert matches_resource("crm.ticket", "crm.ticket")
        assert not matches_resource("crm.ticket", "crm:ticket")
        assert not matches_resource("crm+ticket", "crm:ticket")

    def test_empty_resource(self) -> None:
        assert matches_resource("*", "")
        assert not matches_resource("crm:*", "")


class TestResolveField:
    def test_first_source_wins(self) -> None:
        found, value = resolve_field("x", {"x": 1}, {"x": 2})
        assert (found, value) == (True, 1)

    def test_falls_through_to_later_source(self) -> None:
        found, value = resolve_field("x", {}, {"x": 2})
        assert (found, value) == (True, 2)

    def test_dotted_path(self) -> None:
        found, value = resolve_field("user.role", {"user": {"role": "admin"}})
        assert (found, value) == (True, "admin")

    def test_missing_is_distinguishable_from_none(self) -> None:
        assert resolve_field("x", {"x": None}) == (True, None)
        assert resolve_field("x", {}) == (False, None)


class TestConditions:
    def test_no_conditions_pass(self) -> None:
        assert check_conditions(None, {}) == (True, None)
        assert check_conditions({}, {}) == (True, None)

    def test_equality(self) -> None:
        passed, _ = check_conditions({"status": "open"}, {"status": "open"})
        assert passed
        passed, why = check_conditions({"status": "open"}, {"status": "closed"})
        assert not passed and "status" in why

    def test_membership(self) -> None:
        assert check_conditions({"status": ["open", "pending"]}, {"status": "pending"})[0]
        assert not check_conditions({"status": ["open", "pending"]}, {"status": "closed"})[0]

    def test_operators(self) -> None:
        assert check_conditions({"amount": {"gte": 1000}}, {"amount": 1500})[0]
        assert not check_conditions({"amount": {"gte": 1000}}, {"amount": 500})[0]
        assert check_conditions({"name": {"contains": "prod"}}, {"name": "production-db"})[0]
        assert check_conditions({"env": {"not_in": ["prod"]}}, {"env": "staging"})[0]

    def test_missing_field_fails_closed(self) -> None:
        passed, why = check_conditions({"status": "open"}, {})
        assert not passed
        assert "missing" in why

    def test_type_mismatch_does_not_raise(self) -> None:
        """Comparing a string to an int must fail the condition, not crash the agent."""
        passed, _ = check_conditions({"amount": {"gte": 1000}}, {"amount": "not-a-number"})
        assert not passed

    def test_exists_operator(self) -> None:
        assert check_conditions({"approval": {"exists": True}}, {"approval": "yes"})[0]
        assert check_conditions({"approval": {"exists": False}}, {})[0]
        assert not check_conditions({"approval": {"exists": True}}, {})[0]

    def test_unknown_operator_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown condition operator"):
            check_conditions({"x": {"bogus": 1}}, {"x": 1})
