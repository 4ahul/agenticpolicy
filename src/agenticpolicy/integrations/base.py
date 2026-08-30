"""Framework-agnostic guarding logic.

Every integration is a thin adapter over :class:`ToolGuard`, which owns the
actual sequence — infer what the call is doing, evaluate, run, scan the output,
charge the budget. Keeping that here means the interesting logic is testable
without LangChain, LlamaIndex or LangGraph installed, and the three adapters
stay small enough to read in one sitting.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from typing import Any

from agenticpolicy.core.engine import PolicyEngine
from agenticpolicy.core.policy import Policy
from agenticpolicy.core.types import ActionType, PolicyDecision, ToolCall
from agenticpolicy.exceptions import ApprovalRequired, PolicyViolation

__all__ = ["ToolGuard", "infer_action", "infer_resource", "OnViolation"]

#: What to do when a call is blocked: return a message the model can read and
#: recover from, or raise so the surrounding code handles it.
OnViolation = str  # Literal["block", "raise"]

# Verb fragments in tool names, longest-first so "get_or_create" reads as a
# write rather than a read.
_ACTION_HINTS: list[tuple[ActionType, tuple[str, ...]]] = [
    (ActionType.DELETE, ("delete", "destroy", "remove", "drop", "purge", "revoke", "terminate")),
    (
        ActionType.WRITE,
        (
            "write",
            "create",
            "update",
            "insert",
            "upsert",
            "patch",
            "put",
            "post",
            "set",
            "send",
            "publish",
            "modify",
            "edit",
            "append",
            "save",
            "upload",
            "assign",
        ),
    ),
    (
        ActionType.EXECUTE,
        ("execute", "exec", "run", "deploy", "invoke", "trigger", "call", "shell", "eval"),
    ),
    (
        ActionType.READ,
        (
            "read",
            "get",
            "list",
            "search",
            "query",
            "fetch",
            "find",
            "lookup",
            "describe",
            "view",
            "show",
        ),
    ),
]

_SPLIT = re.compile(r"[_\-.:\s]+")


def infer_action(tool_name: str, explicit: ActionType | str | None = None) -> ActionType:
    """Guess what a tool does from its name.

    ``salesforce_delete_lead`` → ``DELETE``, ``get_ticket`` → ``READ``. Falls
    back to ``EXECUTE``, the most restricted action, so an unrecognized tool
    name fails closed rather than being treated as a harmless read.

    Pass ``explicit`` to skip inference entirely — always do this for tools
    whose names do not describe their effect.
    """
    if explicit is not None:
        return ActionType.coerce(explicit)
    tokens = {t.lower() for t in _SPLIT.split(tool_name) if t}
    for action, hints in _ACTION_HINTS:
        if tokens & set(hints):
            return action
    lowered = tool_name.lower()
    for action, hints in _ACTION_HINTS:
        if any(h in lowered for h in hints):
            return action
    return ActionType.EXECUTE


def infer_resource(tool_name: str, explicit: str | None = None) -> str:
    """Guess the ``provider:type`` resource a tool touches from its name.

    ``salesforce_read_lead`` → ``salesforce:lead``, ``github_pr`` →
    ``github:pr``, ``search`` → ``default:search``. Inference is a convenience
    for prototyping; in production, register resources explicitly via
    ``ToolGuard(resource_map=...)`` so a renamed tool cannot silently change
    which rules apply to it.
    """
    if explicit is not None:
        return explicit
    parts = [p for p in _SPLIT.split(tool_name) if p]
    if not parts:
        return "default:unknown"
    verbs = {h for _, hints in _ACTION_HINTS for h in hints}
    meaningful = [p for p in parts if p.lower() not in verbs] or parts
    if len(meaningful) == 1:
        return f"default:{meaningful[0].lower()}"
    return f"{meaningful[0].lower()}:{'_'.join(m.lower() for m in meaningful[1:])}"


class ToolGuard:
    """Wraps arbitrary callables with policy enforcement.

    Args:
        policy: The policy to enforce. Ignored if ``engine`` is given.
        engine: An existing engine, when several guards should share budget
            state and one audit store.
        context: Baseline context merged into every tool call — put ``user_id``
            and ``task_id`` here once instead of threading them through each
            call site.
        resource_map: Explicit ``{tool_name: "provider:type"}`` overrides.
        action_map: Explicit ``{tool_name: ActionType}`` overrides.
        on_violation: ``"block"`` returns an explanatory string the agent can
            read and route around; ``"raise"`` raises
            :class:`~agenticpolicy.exceptions.PolicyViolation`.

    Example::

        guard = ToolGuard(policy, context={"user_id": "u1"})
        safe_delete = guard.wrap(delete_record, name="crm_delete_record")
        safe_delete(record_id="123")   # -> "[BLOCKED ...]"
    """

    def __init__(
        self,
        policy: Policy | None = None,
        *,
        engine: PolicyEngine | None = None,
        context: dict[str, Any] | None = None,
        resource_map: dict[str, str] | None = None,
        action_map: dict[str, ActionType | str] | None = None,
        on_violation: OnViolation = "block",
    ) -> None:
        if engine is None:
            if policy is None:
                raise ValueError("ToolGuard needs either a policy or an engine")
            engine = PolicyEngine(policy)
        self.engine = engine
        self.policy = engine.policy
        self.context = dict(context or {})
        self.resource_map = dict(resource_map or {})
        self.action_map = {k: ActionType.coerce(v) for k, v in (action_map or {}).items()}
        if on_violation not in ("block", "raise"):
            raise ValueError('on_violation must be "block" or "raise"')
        self.on_violation = on_violation
        self.decisions: list[tuple[ToolCall, PolicyDecision]] = []

    # ------------------------------------------------------------ plumbing

    def build_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> ToolCall:
        """Turn a raw tool invocation into a :class:`ToolCall` for evaluation."""
        return ToolCall(
            agent_id=self.policy.agent_id,
            tool_name=tool_name,
            resource=infer_resource(tool_name, self.resource_map.get(tool_name)),
            action=infer_action(tool_name, self.action_map.get(tool_name)),
            args=args,
            context={**self.context, **(context or {})},
        )

    def _refuse(self, call: ToolCall, decision: PolicyDecision) -> str:
        """Format a refusal the model can act on, or raise."""
        if self.on_violation == "raise":
            exc = ApprovalRequired if decision.requires_approval else PolicyViolation
            raise exc(decision, call)
        label = "NEEDS APPROVAL" if decision.requires_approval else "BLOCKED"
        return f"[{label}] {decision.reason}"

    # ------------------------------------------------------------- wrapping

    def check(self, tool_name: str, args: dict[str, Any], **context: Any) -> PolicyDecision:
        """Evaluate a call without running anything. Useful in tests and dry runs."""
        call = self.build_call(tool_name, args, context=context)
        decision = self.engine.evaluate_sync(call)
        self.decisions.append((call, decision))
        return decision

    def wrap(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        resource: str | None = None,
        action: ActionType | str | None = None,
    ) -> Callable[..., Any]:
        """Return a guarded version of ``fn``.

        Sync and async callables are both supported; the returned wrapper
        matches whichever ``fn`` is. Positional arguments are bound to their
        parameter names before evaluation, so conditions written against
        argument names work regardless of how the tool is called.
        """
        # A lambda or partial may have no usable __name__, so fall back rather
        # than letting an unnamed tool through with an unmappable identity.
        tool_name: str = name or str(getattr(fn, "__name__", None) or "tool")
        if resource is not None:
            self.resource_map[tool_name] = resource
        if action is not None:
            self.action_map[tool_name] = ActionType.coerce(action)

        try:
            signature: inspect.Signature | None = inspect.signature(fn)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            signature = None

        def bind(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
            if signature is None:
                return {"args": list(args), **kwargs}
            try:
                bound = signature.bind_partial(*args, **kwargs)
                return dict(bound.arguments)
            except TypeError:
                return {"args": list(args), **kwargs}

        if inspect.iscoroutinefunction(fn):

            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                call = self.build_call(tool_name, bind(args, kwargs))
                decision = await self.engine.evaluate(call)
                self.decisions.append((call, decision))
                if not decision.allowed:
                    return self._refuse(call, decision)
                result = await fn(*args, **kwargs)
                out_decision, safe = self.engine.check_output(call, result)
                if not out_decision.allowed:
                    # Record the output-stage refusal too, so report() and
                    # .blocked account for data stopped on the way back — not
                    # only calls stopped on the way out.
                    self.decisions.append((call, out_decision))
                    return self._refuse(call, out_decision)
                self.engine.commit(call)
                return safe

            return _copy_meta(async_wrapper, fn, tool_name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            call = self.build_call(tool_name, bind(args, kwargs))
            decision = self.engine.evaluate_sync(call)
            self.decisions.append((call, decision))
            if not decision.allowed:
                return self._refuse(call, decision)
            result = fn(*args, **kwargs)
            out_decision, safe = self.engine.check_output(call, result)
            if not out_decision.allowed:
                self.decisions.append((call, out_decision))
                return self._refuse(call, out_decision)
            self.engine.commit(call)
            return safe

        return _copy_meta(wrapper, fn, tool_name)

    def wrap_all(self, tools: dict[str, Callable[..., Any]]) -> dict[str, Callable[..., Any]]:
        """Guard a whole ``{name: callable}`` registry at once."""
        return {name: self.wrap(fn, name=name) for name, fn in tools.items()}

    # -------------------------------------------------------------- reports

    @property
    def blocked(self) -> list[tuple[ToolCall, PolicyDecision]]:
        """Every call this guard refused, for post-run inspection."""
        return [(c, d) for c, d in self.decisions if not d.allowed]

    def report(self) -> str:
        """A short text summary of what happened during the run."""
        total = len(self.decisions)
        blocked = len(self.blocked)
        lines = [f"{total} tool call(s) evaluated, {blocked} blocked"]
        for call, decision in self.blocked:
            lines.append(
                f"  - {call.tool_name} ({call.action.value} {call.resource}): {decision.reason}"
            )
        return "\n".join(lines)


def _copy_meta(wrapper: Callable[..., Any], fn: Any, name: str) -> Callable[..., Any]:
    """Preserve name and docstring so agent frameworks still see a sane tool."""
    wrapper.__name__ = name
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    return wrapper
