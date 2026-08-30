"""Pre-built policies and the command-line interface."""

from __future__ import annotations

import json

import pytest

from agenticpolicy import Policy, PolicyEngine, PrebuiltRules
from agenticpolicy.cli import main
from agenticpolicy.core.types import Effect
from tests.conftest import make_call


class TestPrebuiltRules:
    def test_catalog_covers_every_factory(self) -> None:
        catalog = PrebuiltRules.catalog()
        assert "least_privilege_support_bot" in catalog
        assert all(doc for doc in catalog.values()), "every prebuilt needs a docstring"

    @pytest.mark.parametrize("name", list(PrebuiltRules.catalog()))
    def test_every_prebuilt_constructs(self, name: str) -> None:
        policy = getattr(PrebuiltRules, name)()
        assert isinstance(policy, Policy)
        assert policy.rules

    def test_support_bot_denies_deletes(self) -> None:
        engine = PolicyEngine(PrebuiltRules.least_privilege_support_bot())
        decision = engine.dry_run(
            make_call("delete", "crm:ticket", context={"ticket_id": "t", "user_id": "u"})
        )
        assert not decision.allowed

    def test_support_bot_blocks_billing_writes(self) -> None:
        engine = PolicyEngine(PrebuiltRules.least_privilege_support_bot())
        decision = engine.dry_run(
            make_call("write", "billing:invoice", context={"ticket_id": "t", "user_id": "u"})
        )
        assert not decision.allowed

    def test_code_agent_blocks_main_branch(self) -> None:
        engine = PolicyEngine(PrebuiltRules.code_agent_with_gates())
        assert engine.dry_run(make_call("write", "github:pr", context={})).allowed
        assert not engine.dry_run(make_call("write", "github:main", context={})).allowed

    def test_code_agent_gates_deploys(self) -> None:
        engine = PolicyEngine(PrebuiltRules.code_agent_with_gates())
        decision = engine.dry_run(make_call("execute", "ci:deploy", context={}))
        assert decision.effect is Effect.NEEDS_APPROVAL

    def test_data_analyst_is_read_only(self) -> None:
        engine = PolicyEngine(PrebuiltRules.data_analyst())
        assert engine.dry_run(make_call("read", "postgres:users", context={})).allowed
        assert not engine.dry_run(make_call("write", "postgres:users", context={})).allowed

    def test_read_only_denies_execute(self) -> None:
        engine = PolicyEngine(PrebuiltRules.read_only())
        assert not engine.dry_run(make_call("execute", "shell:bash", context={})).allowed

    def test_hitl_gates_writes_but_not_reads(self) -> None:
        engine = PolicyEngine(PrebuiltRules.human_in_the_loop())
        assert engine.dry_run(make_call("read", "db:x", context={})).allowed
        assert engine.dry_run(make_call("write", "db:x", context={})).requires_approval

    def test_no_exfiltration_scans_both_directions(self) -> None:
        policy = PrebuiltRules.no_data_exfiltration()
        assert policy.exfil_limits["scan_args"] and policy.exfil_limits["scan_output"]

    def test_prebuilts_are_independent_instances(self) -> None:
        a = PrebuiltRules.read_only()
        b = PrebuiltRules.read_only()
        a.allow("write", ["x:*"])
        assert len(b.allow_rules) == 1


@pytest.fixture
def policy_file(tmp_path):
    path = tmp_path / "policy.json"
    PrebuiltRules.code_agent_with_gates().save(path)
    return str(path)


class TestCLI:
    def test_catalog(self, capsys) -> None:
        assert main(["catalog"]) == 0
        assert "least_privilege_support_bot" in capsys.readouterr().out

    def test_explain(self, capsys, policy_file: str) -> None:
        assert main(["explain", policy_file]) == 0
        out = capsys.readouterr().out
        assert "DENY" in out and "github:main" in out

    def test_test_command_reports_allow(self, capsys, policy_file: str) -> None:
        assert main(["test", policy_file, "write", "github:pr"]) == 0
        assert "ALLOW" in capsys.readouterr().out

    def test_test_command_reports_deny(self, capsys, policy_file: str) -> None:
        main(["test", policy_file, "write", "github:main"])
        assert "DENY" in capsys.readouterr().out

    def test_expect_passes(self, capsys, policy_file: str) -> None:
        assert main(["test", policy_file, "write", "github:main", "--expect", "deny"]) == 0

    def test_expect_fails_with_nonzero_exit(self, policy_file: str) -> None:
        """This is what makes the CLI usable as a CI guard on policy changes."""
        assert main(["test", policy_file, "write", "github:main", "--expect", "allow"]) == 1

    def test_expect_needs_approval(self, policy_file: str) -> None:
        assert (
            main(["test", policy_file, "execute", "ci:deploy", "--expect", "needs_approval"]) == 0
        )

    def test_audit_and_summary(self, capsys, tmp_path) -> None:
        from agenticpolicy import EventStore

        db = tmp_path / "audit.db"
        store = EventStore(db)
        engine = PolicyEngine(PrebuiltRules.read_only(), store=store)
        engine.evaluate_sync(make_call("read", "db:x", agent_id="read_only_agent", context={}))
        engine.evaluate_sync(make_call("delete", "db:x", agent_id="read_only_agent", context={}))
        store.close()

        assert main(["audit", "--db", str(db)]) == 0
        assert "VERDICT" in capsys.readouterr().out

        assert main(["summary", "--db", str(db), "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["total_calls"] == 2 and data["blocked"] == 1

    def test_audit_empty_db(self, capsys, tmp_path) -> None:
        assert main(["audit", "--db", str(tmp_path / "empty.db")]) == 0
        assert "No events" in capsys.readouterr().out

    def test_version(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
