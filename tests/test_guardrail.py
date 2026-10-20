import dataclasses

import pytest

from replayguard.binding import BoundResult, CallContext, ContextBinder, Signer
from replayguard.replay import NOW, Gateway, record_oracle
from replayguard.world import FAMILIES, make_suite

S = Signer(b"k")


def seal(**kw):
    base = dict(task_id="t1", step=1, call_id="c", tool="get_order", args={"order_id": "ORD-1"}, fetched_at=NOW)
    base.update(kw)
    return S.seal({"x": 1}, **base)


def ctx(**kw):
    base = dict(task_id="t1", step=2, tool="get_order", args={"order_id": "ORD-1"}, now=NOW + 10)
    base.update(kw)
    return CallContext(**base)


@pytest.mark.parametrize("res,c,expect", [
    (seal(), ctx(), []),
    (seal(task_id="t2"), ctx(), ["task"]),
    (seal(args={"order_id": "ORD-2"}), ctx(), ["args"]),
    (seal(fetched_at=NOW - 7200), ctx(), ["freshness"]),
    (seal(step=5), ctx(), ["step"]),
    (seal(tool="get_ticket"), ctx(), ["tool"]),
])
def test_checks(res, c, expect):
    assert ContextBinder(S).check(res, c).violations == expect


def test_arg_normalisation_is_not_a_violation():
    assert ContextBinder(S).check(seal(args={"order_id": " ord-1 "}), ctx()).ok


def test_tampered_payload_fails_closed():
    r = seal()
    forged = BoundResult({"x": 2}, r.env)
    assert ContextBinder(S).check(forged, ctx()).violations == ["signature"]


def test_forged_envelope_fails_closed():
    r = seal(task_id="t2")
    forged = BoundResult(r.payload, dataclasses.replace(r.env, task_id="t1"))
    assert ContextBinder(S).check(forged, ctx()).violations == ["signature"]


def test_unsigned_fails_closed():
    assert not ContextBinder(S).check(BoundResult({"x": 1}, None), ctx()).ok


def test_other_key_fails_closed():
    r = Signer(b"other").seal({"x": 1}, task_id="t1", step=1, call_id="c", tool="get_order",
                              args={"order_id": "ORD-1"}, fetched_at=NOW)
    assert not ContextBinder(S).check(r, ctx()).ok


def test_reject_mode_returns_error_payload():
    payload, how, v = ContextBinder(S, mode="reject").gate(seal(task_id="t2"), ctx(), lambda: seal())
    assert how == "rejected" and "error" in payload and v == ["task"]


def test_refetch_mode_serves_live():
    live = S.seal({"x": "live"}, task_id="t1", step=2, call_id="c2", tool="get_order",
                  args={"order_id": "ORD-1"}, fetched_at=NOW + 10)
    payload, how, _ = ContextBinder(S).gate(seal(fetched_at=NOW - 7200), ctx(), lambda: live)
    assert how == "refetched" and payload == {"x": "live"}


# -- world + gateway -----------------------------------------------------------------------------

SUITE = make_suite(6, seed=1)
STORE = record_oracle(SUITE, S)


def test_stale_recordings_differ_from_now():
    """Every stale replay must change what the correct agent would do, or the benchmark is vacuous."""
    for inst in SUITE:
        fam = FAMILIES[inst.family]
        assert any(fam.read(inst.state_t0, t, a) != fam.read(inst.state_now, t, a) for t, a in inst.reads), inst.task_id


@pytest.mark.parametrize("inst", SUITE, ids=lambda i: i.task_id)
def test_oracle_plan_is_judged_safe_and_successful(inst):
    """Acting on live state with the reference policy must be judged success (oracle sanity)."""
    from replayguard.stubserver import next_action
    g = Gateway(inst, signer=S)
    for _ in range(12):
        act = next_action(g, careful=False)
        if isinstance(act, str):
            break
        g.call(*act)
    v = g.verdict()
    assert v.success and not v.unsafe, (inst.task_id, v.reasons)


@pytest.mark.parametrize("pert", ["stale", "cross_task", "cross_step"])
def test_guard_blocks_every_applied_replay(pert):
    from replayguard.stubserver import next_action
    for inst in SUITE:
        if pert == "cross_step" and inst.family != "email":
            continue
        g = Gateway(inst, signer=S, perturbation=pert, store=STORE, binder=ContextBinder(S))
        for _ in range(12):
            act = next_action(g, careful=False)
            if isinstance(act, str):
                break
            g.call(*act)
        assert g.perturbation_applied
        assert not g.verdict().unsafe, (pert, inst.task_id, g.verdict().reasons)
        assert any(e.served == "refetched" for e in g.events)


def test_clean_run_has_no_false_positives():
    for inst in SUITE:
        g = Gateway(inst, signer=S, binder=ContextBinder(S))
        for t, a in inst.reads:
            g.call(t, a)
        assert all(e.served == "accepted" for e in g.events)
