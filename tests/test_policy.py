"""The Policy DSL: construction, validation, serialization, merging."""

from __future__ import annotations

import json

import pytest

from agenticpolicy import Policy
from agenticpolicy.core.types import ActionType, PolicyRule, RuleType
from agenticpolicy.exceptions import PolicyConfigError


class TestConstruction:
    def test_chaining_returns_self(self) -> None:
        policy = Policy("a").allow("read", ["x:*"]).deny("delete", ["*"])
        assert isinstance(policy, Policy)
        assert len(policy.allow_rules) == 1
        assert len(policy.deny_rules) == 1

    def test_rule_ids_are_unique_and_readable(self) -> None:
        policy = Policy().allow("read", ["a:*"]).allow("write", ["b:*"]).deny("delete", ["*"])
        ids = [r.id for r in policy.rules]
        assert len(ids) == len(set(ids))
        assert "allow_0" in ids and "deny_2" in ids

    def test_action_is_case_insensitive(self) -> None:
        policy = Policy().allow("READ", ["x:*"])
        assert policy.allow_rules[0].action is ActionType.READ

    def test_unknown_action_lists_valid_options(self) -> None:
        with pytest.raises(ValueError, match="Valid actions"):
            Policy().allow("frobnicate", ["x:*"])

    def test_string_resources_is_a_helpful_error(self) -> None:
        """`resources="crm:*"` would otherwise iterate character by character."""
        with pytest.raises(PolicyConfigError, match="Did you mean"):
            Policy().allow("read", "crm:*")  # type: ignore[arg-type]

    def test_empty_resources_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty resource list"):
            Policy().allow("read", [])

    def test_require_context_deduplicates(self) -> None:
        policy = Policy().require_context(["a", "b"]).require_context(["b", "c"])
        assert policy.required_context == ["a", "b", "c"]

    def test_invalid_budget_window_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown budget window"):
            Policy().require_approval("execute", ["*"], budget={"per_fortnight": 3})

    def test_negative_budget_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            Policy().require_approval("execute", ["*"], budget={"per_hour": -1})

    def test_rate_limit_needs_a_window(self) -> None:
        with pytest.raises(PolicyConfigError, match="at least one"):
            Policy().rate_limit("execute", ["*"])


class TestExfiltrationConfig:
    def test_defaults_are_populated(self) -> None:
        policy = Policy().prevent_exfiltration()
        assert policy.exfil_limits["max_output_kb"] == 100
        assert len(policy.exfil_limits["block_patterns"]) >= 5

    def test_explicit_empty_patterns_respected(self) -> None:
        """Passing [] means 'size limit only', not 'use the defaults'."""
        policy = Policy().prevent_exfiltration(max_output_kb=5, block_patterns=[])
        assert policy.exfil_limits["block_patterns"] == []
        assert policy.exfil_limits["max_output_kb"] == 5

    def test_zero_kb_is_not_treated_as_unset(self) -> None:
        policy = Policy().prevent_exfiltration(max_output_kb=0)
        assert policy.exfil_limits["max_output_kb"] == 0


class TestSerialization:
    def test_round_trip(self, support_policy: Policy) -> None:
        restored = Policy.from_dict(support_policy.to_dict())
        assert restored.agent_id == support_policy.agent_id
        assert len(restored.allow_rules) == len(support_policy.allow_rules)
        assert len(restored.deny_rules) == len(support_policy.deny_rules)
        assert restored.required_context == support_policy.required_context
        assert restored.exfil_limits == support_policy.exfil_limits

    def test_round_trip_preserves_behaviour(self, support_policy: Policy) -> None:
        from agenticpolicy import PolicyEngine
        from tests.conftest import make_call

        restored = Policy.from_dict(support_policy.to_dict())
        call = make_call("delete", "salesforce:lead")
        assert (
            PolicyEngine(support_policy).dry_run(call).effect
            is PolicyEngine(restored).dry_run(call).effect
        )

    def test_to_dict_is_json_serializable(self, support_policy: Policy) -> None:
        json.dumps(support_policy.to_dict())

    def test_save_and_load(self, support_policy: Policy, tmp_path) -> None:
        path = tmp_path / "policy.json"
        support_policy.save(path)
        assert Policy.load(path).agent_id == support_policy.agent_id

    def test_conditions_survive_round_trip(self) -> None:
        policy = Policy().allow("write", ["crm:ticket"], conditions={"status": ["open"]})
        restored = Policy.from_dict(policy.to_dict())
        assert restored.allow_rules[0].conditions == {"status": ["open"]}

    def test_budget_survives_round_trip(self) -> None:
        policy = Policy().require_approval("execute", ["ci:deploy"], budget={"per_day": 5})
        restored = Policy.from_dict(policy.to_dict())
        assert restored.approve_rules[0].budget == {"per_day": 5}


class TestMerge:
    def test_merge_unions_rules(self) -> None:
        a = Policy("agent").allow("read", ["a:*"])
        b = Policy("agent").allow("read", ["b:*"]).deny("delete", ["*"])
        merged = a.merge(b)
        assert len(merged.allow_rules) == 2
        assert len(merged.deny_rules) == 1

    def test_merge_never_loosens_restrictions(self) -> None:
        a = Policy("x").prevent_exfiltration(max_output_kb=100)
        b = Policy("x").prevent_exfiltration(max_output_kb=10)
        assert a.merge(b).exfil_limits["max_output_kb"] == 10

    def test_merge_unions_required_context(self) -> None:
        a = Policy("x").require_context(["user_id"])
        b = Policy("x").require_context(["ticket_id"])
        assert a.merge(b).required_context == ["ticket_id", "user_id"]

    def test_merge_leaves_originals_untouched(self) -> None:
        a = Policy("x").allow("read", ["a:*"])
        b = Policy("x").allow("read", ["b:*"])
        a.merge(b)
        assert len(a.allow_rules) == 1 and len(b.allow_rules) == 1

    def test_merged_rule_ids_stay_unique(self) -> None:
        a = Policy("x").allow("read", ["a:*"])
        b = Policy("x").allow("read", ["b:*"])
        ids = [r.id for r in a.merge(b).rules]
        assert len(ids) == len(set(ids))


class TestIntrospection:
    def test_explain_mentions_every_rule(self, support_policy: Policy) -> None:
        text = support_policy.explain()
        assert "DENY" in text and "ALLOW" in text and "APPROVE" in text
        assert "EXFIL" in text
        assert "ticket_id" in text

    def test_str_summarises_counts(self, support_policy: Policy) -> None:
        assert "allow=2" in str(support_policy)

    def test_rules_property_includes_everything(self, support_policy: Policy) -> None:
        assert len(support_policy.rules) == (
            len(support_policy.allow_rules)
            + len(support_policy.deny_rules)
            + len(support_policy.approve_rules)
        )


class TestValidation:
    def test_clean_policy_has_no_warnings(self, support_policy: Policy) -> None:
        assert support_policy.validate() == []

    def test_unreachable_approval_gate_is_flagged(self) -> None:
        """Under default-deny, gating an action nothing allows is a silent no-op:
        the policy reads like it gates deploys but actually forbids them."""
        policy = Policy("x").allow("read", ["github:*"])
        policy.require_approval("execute", ["ci:deploy"])
        warnings = policy.validate()
        assert len(warnings) == 1
        assert "unreachable" in warnings[0]
        assert 'policy.allow("execute"' in warnings[0]

    def test_reachable_gate_is_not_flagged(self) -> None:
        policy = Policy("x").allow("execute", ["ci:deploy"])
        policy.require_approval("execute", ["ci:deploy"])
        assert policy.validate() == []

    def test_wildcard_allow_covers_gate(self) -> None:
        policy = Policy("x").allow("*", ["*"])
        policy.rate_limit("execute", ["ci:*"], per_hour=5)
        assert policy.validate() == []

    def test_shadowed_allow_is_flagged(self) -> None:
        policy = Policy("x").allow("delete", ["db:users"]).deny("delete", ["db:users"])
        assert any("shadowed" in w for w in policy.validate())

    def test_zero_output_limit_is_flagged(self) -> None:
        policy = Policy("x").allow("read", ["*"]).prevent_exfiltration(max_output_kb=0)
        assert any("max_output_kb=0" in w for w in policy.validate())

    @pytest.mark.parametrize("name", list(__import__("agenticpolicy").PrebuiltRules.catalog()))
    def test_no_prebuilt_ships_a_warning(self, name: str) -> None:
        from agenticpolicy import PrebuiltRules

        policy = getattr(PrebuiltRules, name)()
        assert policy.validate() == [], f"{name} has policy warnings"

    def test_explain_surfaces_warnings(self) -> None:
        policy = Policy("x").allow("read", ["github:*"])
        policy.require_approval("execute", ["ci:deploy"])
        assert "Warnings:" in policy.explain()


class TestPolicyRule:
    def test_str_is_readable(self) -> None:
        rule = PolicyRule(action=ActionType.READ, resources=["a:*"], rule_type=RuleType.ALLOW)
        assert str(rule) == "allow(read, ['a:*'])"
