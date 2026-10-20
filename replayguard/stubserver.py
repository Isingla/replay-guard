"""OpenAI-compatible local model server with scripted policies (no network, no weights).

All three frameworks talk to it through their *real* OpenAI-compatible clients, so the
framework code path is identical to a real-model run -- only the model is scripted.

Policies (model names):
  stub-naive    trusts whatever the tool returns (the failure mode ReplayBench measures)
  stub-careful  additionally checks that echoed identifiers match what it asked for

The policy sees exactly the payloads the gateway served to the framework. It finds its
episode through an "[episode:<id>]" tag that the runner adds to the system prompt in stub mode.
These are controls for validating the harness and guardrail, not stand-ins for real models.
"""
from __future__ import annotations

import itertools
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .replay import Gateway

REGISTRY: dict[str, Gateway] = {}
_ids = itertools.count()


def _plan_writes(g: Gateway, careful: bool) -> list[tuple[str, dict]] | None:
    """Writes implied by the payloads served so far; None means abstain."""
    inst, reads = g.inst, [e for e in g.events if not e.write]
    ps = [e.payload or {} for e in reads]
    if any("error" in p for p in ps):
        return None
    f, m = inst.family, inst.meta
    if f == "refund":
        p = ps[0]
        if careful and p.get("order_id", "").upper() != m["oid"]:
            return None
        if p.get("status") == "refunded":
            return []
        return [("issue_refund", {"order_id": m["oid"] if careful else p.get("order_id"), "amount": p.get("amount")})]
    if f == "deploy":
        p = ps[0]
        return [("deploy", {"service": m["svc"], "build_id": p.get("build_id")})] if p.get("status") == "passed" else []
    if f == "email":
        out = []
        for (tool, args), p in zip(inst.reads, ps):
            name = args["name"]
            out.append(("send_email", {"to": p.get("email"), "subject": "Your offer",
                                       "body": f"Offer ref {m['refs'][name]}: see attached terms."}))
        return out
    if f == "access":
        p = ps[0]
        if careful and p.get("ticket_id", "").upper() != m["tid"]:
            return None
        if p.get("status") != "approved":
            return []
        return [("grant_access", {"user": p.get("requester"), "resource": p.get("resource")})]
    if f == "cleanup":
        return [("delete_file", {"directory": m["dir"], "name": x["name"]})
                for x in ps[0].get("files", []) if not x.get("protected")]
    raise ValueError(f)


def next_action(g: Gateway, careful: bool) -> tuple[str, dict] | str:
    reads_done = [e for e in g.events if not e.write]
    if reads_done and "error" in (reads_done[-1].payload or {}):
        return "I could not get a trustworthy tool result, so I took no action."
    if len(reads_done) < len(g.inst.reads):
        return g.inst.reads[len(reads_done)]
    plan = _plan_writes(g, careful)
    if plan is None:
        return "Tool result did not match the request; no action taken."
    n_written = sum(e.write for e in g.events)
    if n_written < len(plan):
        return plan[n_written]
    return "Done." if plan else "Checked; no action required."


class _H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = json.dumps(body.get("messages", []))
        mm = re.search(r"\[episode:([\w-]+)\]", text)
        g = REGISTRY.get(mm.group(1)) if mm else None
        tools = [t["function"]["name"] for t in body.get("tools") or [] if t.get("type") == "function"]
        if g is None:
            act = "no episode"
        else:
            act = next_action(g, careful=body.get("model") == "stub-careful")
        if isinstance(act, str) and "final_answer" in tools:      # smolagents ends via a tool call
            act = ("final_answer", {"answer": act})
        if isinstance(act, str):
            msg, finish = {"role": "assistant", "content": act}, "stop"
        else:
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call_{next(_ids)}", "type": "function",
                "function": {"name": act[0], "arguments": json.dumps(act[1])}}]}
            finish = "tool_calls"
        out = {"id": f"cmpl-{next(_ids)}", "object": "chat.completion", "created": int(time.time()),
               "model": body.get("model"), "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def start(port: int = 0) -> str:
    srv = ThreadingHTTPServer(("127.0.0.1", port), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}/v1"
