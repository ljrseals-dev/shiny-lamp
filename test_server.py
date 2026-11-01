"""Client-based tests for Brute Genius using FastMCP's in-memory Client.

Run with:
  pytest test_server.py -v
"""
from __future__ import annotations

import asyncio
import itertools
import json
import time

import pytest
from fastmcp import Client

import enumerator
import xpu as xpu_mod
from mock_provider import MockProvider
from runpod_provider import RunPodProvider

# Import the live server instance
from server import mcp, _plans, provider

SPACE_3X3X3 = {"x": [1, 2, 3], "y": [1, 2, 3], "z": [1, 2, 3]}  # 27 combos
TINY_SPACE = {"a": [0, 1], "b": [0, 1]}  # 4 combos


@pytest.fixture(autouse=True)
def clear_plans():
    """Ensure each test starts with a clean plan store and a healthy balance."""
    _plans.clear()
    provider.balance = 25.0
    provider._nodes.clear()
    provider._stopped.clear()
    yield
    _plans.clear()


@pytest.mark.asyncio
async def test_plan_counts_27():
    async with Client(mcp) as c:
        r = await c.call_tool(
            "brute_genius_plan",
            {
                "space": SPACE_3X3X3,
                "objective": "sphere",
                "max_cost_usd": 10.0,
            },
        )
        data = r.data
        assert data["total_combinations"] == 27
        assert "plan_id" in data
        assert data["state"] == "planned"
        assert data["estimated_cost_usd"] >= 0


@pytest.mark.asyncio
async def test_plan_refuses_over_ceiling():
    """A space whose estimated cost exceeds max_cost_usd must raise and yield no plan_id."""
    async with Client(mcp) as c:
        with pytest.raises(Exception):
            await c.call_tool(
                "brute_genius_plan",
                {
                    "space": {"x": list(range(100_000))},  # huge space
                    "objective": "sphere",
                    "max_cost_usd": 0.0001,  # tiny ceiling
                    "nodes": 1,
                },
            )
        # No plan should have been stored
        assert len(_plans) == 0


@pytest.mark.asyncio
async def test_plan_refuses_over_max_combinations():
    async with Client(mcp) as c:
        with pytest.raises(Exception) as exc_info:
            await c.call_tool(
                "brute_genius_plan",
                {
                    "space": {"x": list(range(10**5)), "y": list(range(10**5))},
                    "objective": "sphere",
                    "max_cost_usd": 1e9,
                },
            )
        assert "exceeds hard limit" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_unknown_objective_rejected():
    async with Client(mcp) as c:
        with pytest.raises(Exception):
            await c.call_tool(
                "brute_genius_plan",
                {
                    "space": TINY_SPACE,
                    "objective": "this_does_not_exist",
                    "max_cost_usd": 10.0,
                },
            )


@pytest.mark.asyncio
async def test_launch_dispatches_correct_shards():
    async with Client(mcp) as c:
        plan = await c.call_tool(
            "brute_genius_plan",
            {
                "space": SPACE_3X3X3,
                "objective": "sphere",
                "max_cost_usd": 10.0,
                "nodes": 3,
            },
        )
        pid = plan.data["plan_id"]
        result = await c.call_tool("brute_genius_launch", {"plan_id": pid})
        assert result.data["state"] == "done"
        assert result.data["combinations_evaluated"] == 27
        # All three shards should be marked done
        status = await c.call_tool("brute_genius_status", {"plan_id": pid})
        assert status.data["shards_done"] == 3
        assert status.data["shards_total"] == 3


@pytest.mark.asyncio
async def test_results_sorted_descending():
    async with Client(mcp) as c:
        plan = await c.call_tool(
            "brute_genius_plan",
            {
                "space": SPACE_3X3X3,
                "objective": "sphere",
                "max_cost_usd": 10.0,
            },
        )
        pid = plan.data["plan_id"]
        await c.call_tool("brute_genius_launch", {"plan_id": pid})
        res = await c.call_tool(
            "brute_genius_results", {"plan_id": pid, "top_k": 5}
        )
        scores = [item["score"] for item in res.data]
        assert scores == sorted(scores, reverse=True)
        assert len(scores) == 5


@pytest.mark.asyncio
async def test_kill_sets_stopped_state():
    async with Client(mcp) as c:
        plan = await c.call_tool(
            "brute_genius_plan",
            {
                "space": SPACE_3X3X3,
                "objective": "sphere",
                "max_cost_usd": 10.0,
                "nodes": 2,
            },
        )
        pid = plan.data["plan_id"]
        # Launch so nodes exist
        await c.call_tool("brute_genius_launch", {"plan_id": pid})
        kill = await c.call_tool("brute_genius_kill", {"plan_id": pid})
        assert kill.data["state"] == "killed"
        status = await c.call_tool("brute_genius_status", {"plan_id": pid})
        assert status.data["state"] == "killed"


@pytest.mark.asyncio
async def test_end_to_end_returns_correct_best_sphere():
    """Sphere is maximised at the origin; with positive integers the best is all 1s."""
    async with Client(mcp) as c:
        plan = await c.call_tool(
            "brute_genius_plan",
            {
                "space": SPACE_3X3X3,
                "objective": "sphere",
                "max_cost_usd": 10.0,
            },
        )
        pid = plan.data["plan_id"]
        await c.call_tool("brute_genius_launch", {"plan_id": pid})
        res = await c.call_tool(
            "brute_genius_results", {"plan_id": pid, "top_k": 1}
        )
        best = res.data[0]
        assert best["combo"] == {"x": 1, "y": 1, "z": 1}
        # Score should be -(1+1+1) = -3
        assert best["score"] == pytest.approx(-3.0)


@pytest.mark.asyncio
async def test_linear_sum_best_is_max_values():
    async with Client(mcp) as c:
        plan = await c.call_tool(
            "brute_genius_plan",
            {
                "space": SPACE_3X3X3,
                "objective": "linear_sum",
                "max_cost_usd": 10.0,
            },
        )
        pid = plan.data["plan_id"]
        await c.call_tool("brute_genius_launch", {"plan_id": pid})
        res = await c.call_tool(
            "brute_genius_results", {"plan_id": pid, "top_k": 1}
        )
        best = res.data[0]
        assert best["combo"] == {"x": 3, "y": 3, "z": 3}
        assert best["score"] == pytest.approx(9.0)


@pytest.mark.asyncio
async def test_xpu_probe_returns_structure():
    async with Client(mcp) as c:
        r = await c.call_tool("xpu_probe", {"chip": "h100"})
        data = r.data
        assert "cuda" in data
        assert "xpu" in data
        assert "mps" in data
        assert "cpu" in data
        assert data["cpu"] is True
        assert "preferred" in data
        assert "provider_rates" in data


@pytest.mark.asyncio
async def test_status_before_launch():
    async with Client(mcp) as c:
        plan = await c.call_tool(
            "brute_genius_plan",
            {
                "space": TINY_SPACE,
                "objective": "sphere",
                "max_cost_usd": 5.0,
            },
        )
        pid = plan.data["plan_id"]
        status = await c.call_tool("brute_genius_status", {"plan_id": pid})
        assert status.data["state"] == "planned"
        assert status.data["combinations_evaluated"] == 0
        assert status.data["best_so_far"] is None


@pytest.mark.asyncio
async def test_nodes_hard_cap():
    async with Client(mcp) as c:
        with pytest.raises(Exception):
            await c.call_tool(
                "brute_genius_plan",
                {
                    "space": TINY_SPACE,
                    "objective": "sphere",
                    "max_cost_usd": 100.0,
                    "nodes": 65,  # exceeds MAX_NODES = 64
                },
            )


@pytest.mark.asyncio
async def test_resources_list_plans():
    async with Client(mcp) as c:
        plan = await c.call_tool(
            "brute_genius_plan",
            {
                "space": TINY_SPACE,
                "objective": "sphere",
                "max_cost_usd": 5.0,
            },
        )
        pid = plan.data["plan_id"]
        # Read the plans resource
        content = await c.read_resource("brutegenius://plans")
        # content is a list of TextResourceContents; just ensure it does not error
        assert content is not None
        # Direct store check
        assert pid in _plans


@pytest.mark.asyncio
async def test_kill_aborts_mid_run(monkeypatch):
    """Kill during launch must abort scoring and preserve partial results.

    Uses a background thread to flip plan state to killed while scoring is
    in a blocking sleep, proving batch_score checks the flag mid-run.
    """
    import threading
    import time as _time
    import xpu as xpu_mod

    def slow_batch_score(shard, fn, device, batch_size=64, plan=None):
        results = []
        batch = []
        for combo in shard:
            if plan is not None and plan.get("state") == "killed":
                raise RuntimeError("Plan killed")
            batch.append(combo)
            if len(batch) >= batch_size:
                if plan is not None and plan.get("state") == "killed":
                    raise RuntimeError("Plan killed")
                _time.sleep(0.05)
                results.extend((c, fn(c)) for c in batch)
                batch = []
        if batch:
            if plan is not None and plan.get("state") == "killed":
                raise RuntimeError("Plan killed")
            results.extend((c, fn(c)) for c in batch)
        return results

    monkeypatch.setattr(xpu_mod, "batch_score", slow_batch_score)

    slow_space = {"x": list(range(1_000)), "y": [1]}
    async with Client(mcp) as c:
        p = await c.call_tool(
            "brute_genius_plan",
            {
                "space": slow_space,
                "objective": "sphere",
                "max_cost_usd": 100.0,
                "nodes": 2,
            },
        )
        pid = p.data["plan_id"]

        def kill_from_thread() -> None:
            _time.sleep(0.08)
            # Directly flip the shared plan state (same process memory).
            # This is what brute_genius_kill does; we bypass the client
            # so the flag is set while the event loop is blocked in sleep.
            plan = _plans.get(pid)
            if plan is not None:
                plan["state"] = "killed"
                plan["killed_at"] = _time.time()
                plan["finished_at"] = plan["killed_at"]

        t = threading.Thread(target=kill_from_thread, daemon=True)
        t.start()

        launch_result = await c.call_tool(
            "brute_genius_launch", {"plan_id": pid}
        )
        t.join(timeout=2.0)

        assert launch_result.data["state"] == "killed"

        status = await c.call_tool("brute_genius_status", {"plan_id": pid})
        assert status.data["state"] == "killed"
        assert status.data.get("killed_at") is not None

        res = await c.call_tool(
            "brute_genius_results", {"plan_id": pid, "top_k": 5}
        )
        assert isinstance(res.data, list)


# ═══════════════════════════════════════════════════════════════════════════
# Hardening pass (A1–A6, B5, E)
# ═══════════════════════════════════════════════════════════════════════════

async def _plan(c, space=SPACE_3X3X3, objective="sphere", max_cost=10.0, **kw):
    r = await c.call_tool(
        "brute_genius_plan",
        {"space": space, "objective": objective, "max_cost_usd": max_cost, **kw},
    )
    return r.data["plan_id"]


def _spent(start: float = 25.0) -> float:
    """Dollars deducted from the mock ledger. (pytest.approx's default tolerance
    on a $25 balance is larger than a typical charge here, so compare exactly.)"""
    return start - provider.balance


async def _wait_for(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.005)


# ─── A1: index-based sharding ──────────────────────────────────────────────

def test_combo_at_matches_product_order():
    space = {"a": [1, 2, 3], "b": ["x", "y"], "c": [0, 1, 2, 3]}  # 24
    expected = [dict(zip(space, t)) for t in itertools.product(*space.values())]
    assert [enumerator.combo_at(space, k) for k in range(24)] == expected
    with pytest.raises(IndexError):
        enumerator.combo_at(space, 24)
    with pytest.raises(IndexError):
        enumerator.combo_at(space, -1)


def test_shards_partition_the_space():
    space = {"a": list(range(3)), "b": list(range(4)), "c": list(range(5))}  # 60
    seen = [tuple(c.values()) for s in enumerator.shard(space, 7) for c in s]
    assert len(seen) == 60
    assert set(seen) == set(itertools.product(*space.values()))


def test_shard_total_work_is_linear(monkeypatch):
    space = {"x": list(range(10)), "y": list(range(10)), "z": list(range(10))}
    calls = {"n": 0}
    real = enumerator.combo_at

    def counting(sp, index):
        calls["n"] += 1
        return real(sp, index)

    monkeypatch.setattr(enumerator, "combo_at", counting)
    per_shard = [sum(1 for _ in s) for s in enumerator.shard(space, 4)]
    assert per_shard == [250, 250, 250, 250]
    assert calls["n"] == 1000  # total, not 4 x total


# ─── A2: cost recomputed at launch ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_launch_override_recomputes_cost():
    async with Client(mcp) as c:
        # 1 x h100 ~ $1.4e-5, under the 2e-5 ceiling; 64 nodes ~ $9e-4, over it.
        pid = await _plan(c, max_cost=2e-5, nodes=1)
        with pytest.raises(Exception) as exc_info:
            await c.call_tool("brute_genius_launch", {"plan_id": pid, "nodes": 64})
        assert "ceiling" in str(exc_info.value).lower()
        assert provider._nodes == {}
        assert _plans[pid]["state"] == "planned"
        assert _plans[pid]["nodes"] == 1  # rejected override left no trace


@pytest.mark.asyncio
async def test_launch_override_within_ceiling_updates_cost():
    async with Client(mcp) as c:
        pid = await _plan(c, max_cost=1.0, nodes=1)
        r = await c.call_tool("brute_genius_launch", {"plan_id": pid, "nodes": 2})
        expected = round(0.01 / 60.0 * 0.084 * 2, 6)
        assert r.data["estimated_cost_usd"] == pytest.approx(expected)
        st = await c.call_tool("brute_genius_status", {"plan_id": pid})
        assert st.data["estimated_cost_usd"] == pytest.approx(expected)
        assert st.data["shards_total"] == 2


@pytest.mark.asyncio
async def test_unknown_chip_rejected_at_plan_and_launch():
    async with Client(mcp) as c:
        with pytest.raises(Exception):
            await _plan(c, chip="quantum9000")
        pid = await _plan(c)
        with pytest.raises(Exception):
            await c.call_tool(
                "brute_genius_launch", {"plan_id": pid, "chip": "quantum9000"}
            )
        assert provider._nodes == {}


# ─── A3: balance is a real ledger ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_launch_refuses_when_balance_too_low():
    big = {"x": list(range(1000)), "y": list(range(1000))}  # 1e6 combos ~ $0.014
    async with Client(mcp) as c:
        pid = await _plan(c, space=big, max_cost=10.0)
        provider.balance = 0.005
        with pytest.raises(Exception) as exc_info:
            await c.call_tool("brute_genius_launch", {"plan_id": pid})
        assert "balance" in str(exc_info.value).lower()
        assert _plans[pid]["state"] == "planned"
        assert provider._nodes == {}
        assert provider.balance == 0.005


@pytest.mark.asyncio
async def test_launch_charges_balance():
    async with Client(mcp) as c:
        pid = await _plan(c)
        cost = _plans[pid]["est_cost_usd"]
        assert cost > 0
        await c.call_tool("brute_genius_launch", {"plan_id": pid})
        assert _spent() > 0
        assert abs(_spent() - cost) < 1e-9


@pytest.mark.asyncio
async def test_mock_charge_rejects_overdraft():
    p = MockProvider(balance=1.0)
    assert await p.charge(0.25) == pytest.approx(0.75)
    with pytest.raises(RuntimeError):
        await p.charge(5.0)
    assert p.balance == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_runpod_charge_is_not_silent():
    rp = object.__new__(RunPodProvider)  # skip __init__: no SDK / API key here
    with pytest.raises(NotImplementedError):
        await rp.charge(1.0)


# ─── A4: relaunch is refused ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_relaunch_done_plan_rejected():
    async with Client(mcp) as c:
        pid = await _plan(c, nodes=3)
        await c.call_tool("brute_genius_launch", {"plan_id": pid})
        with pytest.raises(Exception) as exc_info:
            await c.call_tool("brute_genius_launch", {"plan_id": pid})
        assert "cannot be launched" in str(exc_info.value).lower()
        st = await c.call_tool("brute_genius_status", {"plan_id": pid})
        assert st.data["shards_done"] == st.data["shards_total"] == 3


@pytest.mark.asyncio
async def test_relaunch_killed_plan_rejected():
    async with Client(mcp) as c:
        pid = await _plan(c, nodes=2)
        await c.call_tool("brute_genius_launch", {"plan_id": pid})
        await c.call_tool("brute_genius_kill", {"plan_id": pid})
        balance = provider.balance
        with pytest.raises(Exception) as exc_info:
            await c.call_tool("brute_genius_launch", {"plan_id": pid})
        assert "create a new plan" in str(exc_info.value).lower()
        assert provider.balance == balance  # no second charge


# ─── A5: live partials + kill is reachable mid-run ─────────────────────────

@pytest.mark.asyncio
async def test_kill_tool_reports_partial_results(monkeypatch):
    def slow_batch_score(shard, fn, device, batch_size=64, plan=None):
        results, batch = [], []

        def _check():
            if plan is not None and plan.get("state") == "killed":
                raise RuntimeError("Plan killed")

        for combo in shard:
            _check()
            batch.append(combo)
            if len(batch) >= batch_size:
                _check()
                time.sleep(0.05)  # runs in a worker thread, loop stays free
                results.extend((c, fn(c)) for c in batch)
                batch = []
        if batch:
            _check()
            results.extend((c, fn(c)) for c in batch)
        return results

    monkeypatch.setattr(xpu_mod, "batch_score", slow_batch_score)

    async with Client(mcp) as c:
        pid = await _plan(c, space={"x": list(range(800)), "y": [1]}, max_cost=100.0, nodes=4)
        launch_task = asyncio.create_task(
            c.call_tool("brute_genius_launch", {"plan_id": pid})
        )
        # Wait until at least one shard has landed, then kill via the real tool.
        await _wait_for(lambda: _plans[pid]["combinations_evaluated"] > 0)
        kill = await c.call_tool("brute_genius_kill", {"plan_id": pid})
        launched = await launch_task

        assert kill.data["state"] == "killed"
        assert kill.data["partial_results"] > 0
        assert kill.data["combinations_evaluated"] > 0
        assert kill.data["nodes_stopped"] == 4
        assert launched.data["state"] == "killed"
        assert launched.data["combinations_evaluated"] < 800
        assert _plans[pid]["best_so_far"] is not None
        assert all(n["status"] == "stopped" for n in provider._nodes.values())

        res = await c.call_tool("brute_genius_results", {"plan_id": pid, "top_k": 3})
        scores = [r["score"] for r in res.data]
        assert scores == sorted(scores, reverse=True) and len(scores) == 3


@pytest.mark.asyncio
async def test_results_available_between_shards(monkeypatch):
    """brute_genius_status/results show real data while a run is still live."""
    gate = {"release": False}

    def gated(shard, fn, device, batch_size=64, plan=None):
        out = [(c, fn(c)) for c in shard]
        if plan["shards"] and plan["shards"][-1]["status"] == "running" and not any(
            s["status"] == "done" for s in plan["shards"]
        ):
            pass  # first shard returns immediately
        else:
            t0 = time.monotonic()
            while not gate["release"] and time.monotonic() - t0 < 5:
                time.sleep(0.005)
        return out

    monkeypatch.setattr(xpu_mod, "batch_score", gated)
    async with Client(mcp) as c:
        pid = await _plan(c, nodes=3)
        task = asyncio.create_task(c.call_tool("brute_genius_launch", {"plan_id": pid}))
        await _wait_for(lambda: _plans[pid]["combinations_evaluated"] > 0)
        st = await c.call_tool("brute_genius_status", {"plan_id": pid})
        assert st.data["state"] == "running"
        assert st.data["best_so_far"] is not None
        live = await c.call_tool("brute_genius_results", {"plan_id": pid, "top_k": 2})
        assert len(live.data) == 2
        gate["release"] = True
        done = await task
        assert done.data["state"] == "done"
        assert done.data["combinations_evaluated"] == 27


# ─── A6: provisioning rolls back ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_provisioning_rollback_on_partial_failure(monkeypatch):
    real = MockProvider.create_instance
    calls = {"n": 0}

    async def flaky(self, name, chip):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom")
        return await real(self, name, chip)

    monkeypatch.setattr(MockProvider, "create_instance", flaky)
    async with Client(mcp) as c:
        pid = await _plan(c, nodes=4)
        with pytest.raises(Exception) as exc_info:
            await c.call_tool("brute_genius_launch", {"plan_id": pid})
        assert "boom" in str(exc_info.value)
        assert len(provider._nodes) == 2
        assert all(n["status"] == "stopped" for n in provider._nodes.values())
        assert _plans[pid]["state"] == "planned"
        assert _plans[pid]["shards"] == []
        assert provider.balance == 25.0  # nothing charged

        # A "planned" plan may be relaunched, starting from clean counters.
        r = await c.call_tool("brute_genius_launch", {"plan_id": pid})
        assert r.data["state"] == "done"
        assert r.data["combinations_evaluated"] == 27
        assert len(_plans[pid]["shards"]) == 4
        assert abs(_spent() - _plans[pid]["est_cost_usd"]) < 1e-9


@pytest.mark.asyncio
async def test_kill_during_provisioning_aborts_launch(monkeypatch):
    real = MockProvider.create_instance
    calls = {"n": 0}

    async def kill_midway(self, name, chip):
        calls["n"] += 1
        out = await real(self, name, chip)
        if calls["n"] == 2:
            for p in _plans.values():
                p["state"] = "killed"  # what brute_genius_kill does first
        return out

    monkeypatch.setattr(MockProvider, "create_instance", kill_midway)
    async with Client(mcp) as c:
        pid = await _plan(c, nodes=3)
        with pytest.raises(Exception):
            await c.call_tool("brute_genius_launch", {"plan_id": pid})
        assert _plans[pid]["state"] == "killed"
        assert all(n["status"] == "stopped" for n in provider._nodes.values())
        assert provider.balance == 25.0


@pytest.mark.asyncio
async def test_scoring_error_stops_nodes_and_marks_failed():
    async with Client(mcp) as c:
        pid = await _plan(c, space={"x": ["not-a-number", "also-not"]}, objective="linear_sum", nodes=2)
        with pytest.raises(Exception):
            await c.call_tool("brute_genius_launch", {"plan_id": pid})
        assert _plans[pid]["state"] == "failed"
        assert len(provider._nodes) == 2
        assert all(n["status"] == "stopped" for n in provider._nodes.values())
        with pytest.raises(Exception):  # a failed (already charged) plan is not retried
            await c.call_tool("brute_genius_launch", {"plan_id": pid})


# ─── B5: pricing resource reads the ledger via the async API ───────────────

@pytest.mark.asyncio
async def test_pricing_resource_async_or_dropped(monkeypatch):
    async def fake_balance():
        return 12.5

    monkeypatch.setattr(provider, "get_balance", fake_balance)
    async with Client(mcp) as c:
        content = await c.read_resource("brutegenius://pricing")
        payload = json.loads(content[0].text)
        assert payload["balance_usd"] == 12.5
        assert "h100" in payload["rates_usd_per_hour"]


# ─── E: small correctness fixes ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_space_rejected():
    async with Client(mcp) as c:
        with pytest.raises(Exception) as exc_info:
            await _plan(c, space={})
        assert "at least one axis" in str(exc_info.value).lower()
        assert len(_plans) == 0


@pytest.mark.asyncio
async def test_launch_uses_chip_device_hints(monkeypatch):
    assert xpu_mod.CHIP_DEVICE_HINTS["cpu"] == "cpu"
    assert xpu_mod.CHIP_DEVICE_HINTS["h100"] == "cuda"
    seen = []
    real = xpu_mod.pick_device

    def spy(preferred="cuda"):
        seen.append(preferred)
        return real(preferred)

    monkeypatch.setattr(xpu_mod, "pick_device", spy)
    async with Client(mcp) as c:
        for chip in ("a100", "cpu"):
            pid = await _plan(c, chip=chip)
            await c.call_tool("brute_genius_launch", {"plan_id": pid})
    assert seen == ["cuda", "cpu"]
