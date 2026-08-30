"""Code agent: opens PRs freely, needs sign-off to deploy.

Shows the three controls that matter for a coding agent — protected branches,
approval gates, and a deploy budget that survives a retry loop.

    python examples/code_agent.py
"""

from __future__ import annotations

from agenticpolicy import PolicyEngine, PrebuiltRules
from agenticpolicy.integrations.base import ToolGuard


def read_github_repo(path: str) -> str:
    return f"contents of {path}"


def write_github_pr(title: str, body: str) -> str:
    return f"Opened PR: {title}"


def write_github_main(message: str) -> str:
    return f"Pushed directly to main: {message}"


def execute_ci_deploy(environment: str) -> str:
    return f"Deployed to {environment}"


def main() -> None:
    # Two deploys a day, so the third attempt is refused even with approval.
    policy = PrebuiltRules.code_agent_with_gates(deploys_per_day=2)
    print(policy.explain())
    print()

    guard = ToolGuard(
        engine=PolicyEngine(policy),
        resource_map={
            "read_github_repo": "github:repo",
            "write_github_pr": "github:pr",
            "write_github_main": "github:main",
            "execute_ci_deploy": "ci:deploy",
        },
    )
    tools = guard.wrap_all(
        {
            "read_github_repo": read_github_repo,
            "write_github_pr": write_github_pr,
            "write_github_main": write_github_main,
            "execute_ci_deploy": execute_ci_deploy,
        }
    )

    print("1. Read the codebase")
    print("  ", tools["read_github_repo"](path="src/app.py"))

    print("\n2. Open a pull request")
    print("  ", tools["write_github_pr"](title="Fix retry backoff", body="..."))

    print("\n3. Push straight to main — the protected branch")
    print("  ", tools["write_github_main"](message="hotfix"))

    print("\n4. Deploy — gated, so it stops for a human")
    print("  ", tools["execute_ci_deploy"](environment="production"))

    # Approval gates hold the call rather than failing it. Once a human signs
    # off, the same call is re-evaluated against a policy that permits it.
    print("\n5. After sign-off: re-run under an approved policy")
    approved = PrebuiltRules.code_agent_with_gates(deploys_per_day=2)
    approved.approve_rules[0].limits = {"approval_exempt": 1}  # budget still applies
    approved_guard = ToolGuard(
        engine=PolicyEngine(approved),
        resource_map={"execute_ci_deploy": "ci:deploy"},
    )
    deploy = approved_guard.wrap(execute_ci_deploy, name="execute_ci_deploy")
    for attempt in range(1, 4):
        print(f"   attempt {attempt}: {deploy(environment='production')}")

    print("\n" + approved_guard.report())
    print("\nThe third deploy is refused by the daily budget — a retry loop cannot")
    print("spend more than the policy allows, which is the point of the cap.")


if __name__ == "__main__":
    main()
