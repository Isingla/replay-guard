"""Reference guardrail: contextual binding of tool results.

Every tool result produced by the real executor is wrapped in a signed Envelope that
binds it to (task, step, tool, args, freshness). Before a result -- live, cached or
replayed -- is handed to the model, ContextBinder verifies it against the *current*
call context. On mismatch it either re-fetches live ("refetch") or returns a
structured error ("reject"). Results without a valid signature fail closed.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

ALL_CHECKS = ("signature", "tool", "task", "args", "step", "freshness")


def canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def norm_args(args: dict) -> dict:
    return {k: (v.strip().lower() if isinstance(v, str) else v) for k, v in sorted(args.items())}


def args_hash(args: dict) -> str:
    return hashlib.sha256(canon(norm_args(args)).encode()).hexdigest()[:16]


@dataclass
class Envelope:
    task_id: str
    step: int
    call_id: str
    tool: str
    args_hash: str
    fetched_at: float
    payload_hash: str
    sig: str = ""

    def body(self) -> dict:
        d = asdict(self)
        d.pop("sig")
        return d


@dataclass
class BoundResult:
    payload: dict
    env: Envelope | None

    def to_json(self) -> dict:
        return {"payload": self.payload, "env": asdict(self.env) if self.env else None}

    @staticmethod
    def from_json(d: dict) -> "BoundResult":
        return BoundResult(d["payload"], Envelope(**d["env"]) if d.get("env") else None)


class Signer:
    def __init__(self, key: bytes):
        self.key = key

    def seal(self, payload: dict, *, task_id: str, step: int, call_id: str, tool: str, args: dict,
             fetched_at: float) -> BoundResult:
        env = Envelope(task_id, step, call_id, tool, args_hash(args), fetched_at,
                       hashlib.sha256(canon(payload).encode()).hexdigest()[:16])
        env.sig = self._mac(env)
        return BoundResult(payload, env)

    def _mac(self, env: Envelope) -> str:
        return hmac.new(self.key, canon(env.body()).encode(), hashlib.sha256).hexdigest()

    def verify(self, r: BoundResult) -> bool:
        if r.env is None:
            return False
        ph = hashlib.sha256(canon(r.payload).encode()).hexdigest()[:16]
        return ph == r.env.payload_hash and hmac.compare_digest(r.env.sig, self._mac(r.env))


@dataclass
class CallContext:
    task_id: str
    step: int
    tool: str
    args: dict
    now: float


@dataclass
class Decision:
    ok: bool
    violations: list[str] = field(default_factory=list)


class ContextBinder:
    """Verifies a candidate result against the current call context.

    ttl_s: per-tool freshness budget (seconds); default applies to tools not listed.
    mode:  "refetch" -> on violation, call `fetch_live()` and serve that instead
           "reject"  -> on violation, serve a structured error to the model
    """

    def __init__(self, signer: Signer, *, ttl_s: dict[str, float] | None = None, default_ttl_s: float = 300,
                 mode: str = "refetch", checks: tuple[str, ...] = ALL_CHECKS):
        assert mode in ("refetch", "reject")
        self.signer, self.ttl, self.default_ttl = signer, ttl_s or {}, default_ttl_s
        self.mode, self.checks = mode, set(checks)

    def check(self, r: BoundResult, ctx: CallContext) -> Decision:
        v: list[str] = []
        c = self.checks
        if "signature" in c and not self.signer.verify(r):
            return Decision(False, ["signature"])      # nothing else in the envelope can be trusted
        e = r.env
        if e is None:
            return Decision(False, ["unbound"])
        if "tool" in c and e.tool != ctx.tool:
            v.append("tool")
        if "task" in c and e.task_id != ctx.task_id:
            v.append("task")
        if "args" in c and e.args_hash != args_hash(ctx.args):
            v.append("args")
        if "step" in c and e.step > ctx.step:
            v.append("step")
        if "freshness" in c and ctx.now - e.fetched_at > self.ttl.get(ctx.tool, self.default_ttl):
            v.append("freshness")
        return Decision(not v, v)

    def gate(self, candidate: BoundResult, ctx: CallContext,
             fetch_live: Callable[[], BoundResult]) -> tuple[dict, str, list[str]]:
        """Returns (payload to show the model, how it was served, violations)."""
        d = self.check(candidate, ctx)
        if d.ok:
            return candidate.payload, "accepted", []
        if self.mode == "refetch":
            fresh = fetch_live()
            assert self.check(fresh, ctx).ok, "live result failed its own binding"
            return fresh.payload, "refetched", d.violations
        return ({"error": "tool_result_rejected",
                 "detail": f"result not bound to this call ({', '.join(d.violations)}); it may be stale or "
                           f"from another task. Do not act on it."}, "rejected", d.violations)
