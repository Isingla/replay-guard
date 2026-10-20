"""Recording + replay at the tool boundary.

Gateway is the single choke point every framework's tools call into. It
  1. executes reads against the live world (state_now) and seals them,
  2. optionally swaps in a recorded result (the perturbation),
  3. optionally runs the ContextBinder guardrail on whatever would be served,
  4. executes writes against the world and logs them for the oracle.

Perturbations (applied to read tools only):
  cross_task  -- a fresh (60s old) recording of the same tool from a *different* task
  stale       -- this task's own recording of the same call, made at t0 (2h ago) before the world changed
  cross_step  -- an earlier result from this episode for *different* args of the same tool
"""
from __future__ import annotations

import hashlib
import itertools
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .binding import BoundResult, CallContext, ContextBinder, Signer, args_hash
from .world import DONOR_AGE_S, FAMILIES, STALE_AGE_S, Instance, Verdict, is_write

NOW = 1_800_000_000.0
PERTURBATIONS = ("clean", "cross_task", "stale", "cross_step")


class TraceStore:
    """Recorded agent traces: every read result an agent saw, with its envelope.

    Indexed by (task_id, tool, args_hash, kind), kind in {"t0", "recent"}.
    """

    def __init__(self):
        self.rows: list[dict] = []
        self._idx: dict[tuple, BoundResult] = {}
        self._by_family_tool: dict[tuple, list[tuple[str, BoundResult]]] = {}
        self._lock = threading.Lock()

    def add(self, family: str, kind: str, r: BoundResult, args: dict):
        with self._lock:
            k = (r.env.task_id, r.env.tool, r.env.args_hash, kind)
            if k in self._idx:
                return
            self._idx[k] = r
            self._by_family_tool.setdefault((family, r.env.tool, kind), []).append((r.env.task_id, r))
            self.rows.append({"family": family, "kind": kind, "args": args, **r.to_json()})

    def get(self, task_id, tool, args, kind) -> BoundResult | None:
        return self._idx.get((task_id, tool, args_hash(args), kind))

    def donor(self, family, tool, task_id) -> BoundResult | None:
        cands = sorted([x for x in self._by_family_tool.get((family, tool, "recent"), []) if x[0] != task_id],
                       key=lambda x: x[0])
        if not cands:
            return None
        h = int(hashlib.sha256(task_id.encode()).hexdigest(), 16)
        return cands[h % len(cands)][1]

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in self.rows))

    @staticmethod
    def load(path: Path) -> "TraceStore":
        s = TraceStore()
        for line in path.read_text().splitlines():
            d = json.loads(line)
            s.add(d["family"], d["kind"], BoundResult.from_json(d), d["args"])
        return s


@dataclass
class Event:
    step: int
    tool: str
    args: dict
    write: bool
    served: str = "live"                 # live | replayed | accepted | refetched | rejected
    perturbed: bool = False
    violations: list[str] = field(default_factory=list)
    payload: dict | None = None


_call_ids = itertools.count()


class Gateway:
    def __init__(self, inst: Instance, *, signer: Signer, perturbation: str = "clean",
                 store: TraceStore | None = None, binder: ContextBinder | None = None,
                 state: str = "now", clock: float = NOW, record_into: TraceStore | None = None,
                 record_kind: str = "recent"):
        assert perturbation in PERTURBATIONS
        self.inst, self.fam = inst, FAMILIES[inst.family]
        self.signer, self.pert, self.store, self.binder = signer, perturbation, store, binder
        self.state = inst.state_now if state == "now" else inst.state_t0
        self.clock = clock
        self.record_into, self.record_kind = record_into, record_kind
        self.step = 0
        self.events: list[Event] = []
        self.writes: list[tuple[str, dict]] = []
        self._served_reads: list[tuple[dict, BoundResult]] = []
        self._perturbed_once = False
        self._lock = threading.Lock()

    # -- the one function every framework tool calls -------------------------------------------
    def call(self, tool: str, args: dict) -> str:
        with self._lock:
            self.step += 1
            args = {k: v for k, v in args.items() if v is not None}
            if is_write(self.inst.family, tool):
                self.writes.append((tool, args))
                self.events.append(Event(self.step, tool, args, True, "live"))
                return json.dumps({"ok": True})
            live = self._live(tool, args)
            if self.record_into is not None and "error" not in live.payload:
                self.record_into.add(self.inst.family, self.record_kind, live, args)
            cand, perturbed = self._candidate(tool, args), False
            if cand is not None:
                perturbed = True
            served_r = cand or live
            ev = Event(self.step, tool, args, False, "replayed" if perturbed else "live", perturbed)
            if self.binder is not None:
                ctx = CallContext(self.inst.task_id, self.step, tool, args, self.clock)
                payload, how, viol = self.binder.gate(served_r, ctx, lambda: live)
                ev.served, ev.violations = how, viol
            else:
                payload = served_r.payload
            ev.payload = payload
            self.events.append(ev)
            self._served_reads.append((args, served_r))
            return json.dumps(payload)

    def _live(self, tool, args) -> BoundResult:
        return self.signer.seal(self.fam.read(self.state, tool, args), task_id=self.inst.task_id,
                                step=self.step, call_id=f"c{next(_call_ids)}", tool=tool, args=args,
                                fetched_at=self.clock)

    def _candidate(self, tool, args) -> BoundResult | None:
        if self.pert == "clean" or self._perturbed_once:
            return None
        r = None
        if self.pert == "stale":
            r = self.store.get(self.inst.task_id, tool, args, "t0")
        elif self.pert == "cross_task":
            r = self.store.donor(self.inst.family, tool, self.inst.task_id)
        elif self.pert == "cross_step":
            ah = args_hash(args)
            prev = [br for a, br in self._served_reads if br.env and br.env.tool == tool and args_hash(a) != ah]
            r = prev[0] if prev else None
        if r is not None:
            self._perturbed_once = True
        return r

    # -- results ---------------------------------------------------------------------------------
    @property
    def perturbation_applied(self) -> bool:
        return any(e.perturbed for e in self.events)

    def verdict(self) -> Verdict:
        return self.fam.judge(self.inst, self.writes)


def record_oracle(insts: list[Instance], signer: Signer) -> TraceStore:
    """Fallback recorder: executes each task's reference reads (identical results to what an agent sees,
    since read tools are deterministic in (state, args)). Prefer `record` with a real agent."""
    store = TraceStore()
    for inst in insts:
        for kind, state, clock in (("t0", "t0", NOW - STALE_AGE_S), ("recent", "now", NOW - DONOR_AGE_S)):
            g = Gateway(inst, signer=signer, state=state, clock=clock, record_into=store, record_kind=kind)
            for tool, args in inst.reads:
                g.call(tool, args)
    return store
