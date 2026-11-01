"""Brute Genius MCP Server.

Exhaustive combinatorial search over a discrete space. The space is sharded and
the shards are scored one after another, in-process, on the host (in a worker
thread so the server stays responsive to status/kill). Fleet nodes are
provisioned and billed through a GPUProvider seam, but remote execution is not
implemented yet; see the README Roadmap.

Safety rules (each covered by a test in test_server.py):
  - plan() refuses if estimated cost > max_cost_usd, and stores that ceiling
  - launch() recomputes the cost after any chip/nodes override and refuses if it
    exceeds the plan's ceiling, before any node is created
  - launch() refuses if the estimate exceeds the provider balance, then charges
    the estimate up front (provider.charge) before scoring. Insufficient funds
    raise. There is no refund on early kill yet
  - a launch that fails while provisioning or scoring stops every node it created
  - a killed or failed plan cannot be relaunched; create a new plan
  - kill() is reachable while scoring runs and stops every node for a plan
  - Hard caps: MAX_NODES = 64, MAX_COMBINATIONS = 10**9
  - Objective functions are looked up by name from a registry only
  - Never write to stdout in stdio mode; use ctx logging / stderr
"""
from __future__ import annotations

import asyncio
import sys
import time
import uuid
from typing import Any

from fastmcp import Context, FastMCP

import enumerator
import objectives
import xpu
from mock_provider import MockProvider

# ─── Hard safety limits ────────────────────────────────────────────────────
MAX_NODES = 64
MAX_COMBINATIONS = enumerator.MAX_COMBINATIONS

# Rough planning heuristic: combinations scored per node per minute.
# Tune against real objective cost before trusting the ceiling in production.
COMBOS_PER_NODE_PER_MINUTE = 100_000

mcp = FastMCP(
    "brute-genius",
    instructions=(
        "Brute Genius performs exhaustive Cartesian search over a discrete "
        "space. Shards are currently scored sequentially on the host. Always "
        "call brute_genius_plan first and respect the cost ceiling. Never "
        "launch without a plan_id."
    ),
)

provider = MockProvider(balance=25.0)

# In-memory plan store. Keys are plan_id strings.
_plans: dict[str, dict[str, Any]] = {}


def _log(msg: str) -> None:
    """Write to stderr so we never pollute stdout in stdio transport."""
    print(msg, file=sys.stderr, flush=True)


def _estimate(total: int, nodes: int, chip: str) -> tuple[float, float]:
    """Return (estimated minutes, estimated USD) for a run. One formula, used by
    both plan() and launch() so an override can never dodge the ceiling."""
    rate = provider.hourly_rate(chip)
    est_minutes = max(total / (nodes * COMBOS_PER_NODE_PER_MINUTE), 0.01)
    est_cost = est_minutes / 60.0 * rate * nodes
    return est_minutes, est_cost


async def _stop_nodes(names: list[str]) -> None:
    """Best-effort stop of every named node; failures are logged, not hidden."""
    for name in names:
        try:
            await provider.stop_instance(name)
        except Exception as exc:
            _log(f"failed to stop node {name}: {exc!r}")


# ─── TOOLS ─────────────────────────────────────────────────────────────────

@mcp.tool()
async def brute_genius_plan(
    space: dict[str, list[Any]],
    objective: str,
    max_cost_usd: float,
    chip: str = "h100",
    nodes: int = 1,
) -> dict[str, Any]:
    """Enumerate the full Cartesian product and return a cost estimate.

    Does NOT execute anything. Refuses to return a plan_id when the
    estimated cost exceeds max_cost_usd.

    Args:
        space: Mapping of axis name → list of discrete values.
        objective: Name of a registered objective (see objectives registry).
        max_cost_usd: Hard ceiling in USD. Plan is refused above this.
        chip: Accelerator type (h100, a100, l40s, cpu).
        nodes: Number of nodes to shard across (1–64).
    """
    if not space:
        raise ValueError("Search space must contain at least one axis")

    # Validate objective name early (raises KeyError if unknown)
    objectives.get(objective)

    total = enumerator.count(space)
    if total == 0:
        raise ValueError("Search space is empty (at least one axis has no values)")
    if total > MAX_COMBINATIONS:
        raise ValueError(
            f"Space has {total:,} combinations, exceeds hard limit of "
            f"{MAX_COMBINATIONS:,}"
        )
    if nodes < 1:
        raise ValueError("nodes must be >= 1")
    if nodes > MAX_NODES:
        raise ValueError(f"nodes={nodes} exceeds MAX_NODES={MAX_NODES}")

    chip = chip.lower()
    if chip not in provider.RATES:
        raise ValueError(f"Unknown chip '{chip}'. Known: {sorted(provider.RATES)}")
    est_minutes, est_cost = _estimate(total, nodes, chip)

    if est_cost > max_cost_usd:
        raise ValueError(
            f"Estimated cost ${est_cost:.4f} exceeds ceiling "
            f"${max_cost_usd:.2f}. Reduce space, increase nodes, "
            f"or raise max_cost_usd."
        )

    plan_id = str(uuid.uuid4())
    _plans[plan_id] = {
        "plan_id": plan_id,
        "space": space,
        "objective": objective,
        "chip": chip,
        "nodes": nodes,
        "total": total,
        "max_cost_usd": max_cost_usd,
        "est_cost_usd": round(est_cost, 6),
        "est_minutes": round(est_minutes, 4),
        "state": "planned",
        "shards": [],          # filled on launch
        "results": [],         # list of (combo, score)
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "killed_at": None,
        "combinations_evaluated": 0,
        "best_so_far": None,
        "device_used": None,
    }

    return {
        "plan_id": plan_id,
        "total_combinations": total,
        "estimated_cost_usd": round(est_cost, 4),
        "estimated_minutes": round(est_minutes, 4),
        "chip": chip,
        "nodes": nodes,
        "state": "planned",
        "objective": objective,
    }


@mcp.tool()
async def brute_genius_launch(
    plan_id: str,
    ctx: Context,
    chip: str | None = None,
    nodes: int | None = None,
) -> dict[str, Any]:
    """Shard the enumerated space and score every combination.

    Before any node is created this recomputes the cost for the final chip and
    node count (refusing anything above the plan's ceiling) and checks it
    against the provider balance. After provisioning it charges the estimate up
    front; there is no refund if the run is killed early. Shards are then
    scored one after another on the host (in a worker thread, so status and
    kill stay responsive). A launch that fails stops every node it created.
    Only a "planned" plan can be launched.

    Args:
        plan_id: Identifier returned by brute_genius_plan.
        chip: Optional override of the chip chosen at plan time.
        nodes: Optional override of the node count chosen at plan time.
    """
    plan = _plans.get(plan_id)
    if plan is None:
        raise ValueError(f"Unknown plan_id '{plan_id}'")
    if plan["state"] == "killed":
        raise ValueError(f"Plan '{plan_id}' was killed; create a new plan")
    if plan["state"] != "planned":
        raise ValueError(
            f"Plan '{plan_id}' is in state '{plan['state']}' and cannot be launched"
        )

    # Resolve overrides locally. Nothing is written to the plan until every
    # check below has passed, so a refused launch leaves the plan untouched.
    new_chip = chip.lower() if chip is not None else plan["chip"]
    new_nodes = nodes if nodes is not None else plan["nodes"]
    if new_chip not in provider.RATES:
        raise ValueError(f"Unknown chip '{new_chip}'. Known: {sorted(provider.RATES)}")
    if new_nodes < 1 or new_nodes > MAX_NODES:
        raise ValueError(f"nodes must be 1–{MAX_NODES}")

    # Safety: recompute cost for the final configuration and enforce the ceiling.
    est_minutes, est_cost = _estimate(plan["total"], new_nodes, new_chip)
    if est_cost > plan["max_cost_usd"]:
        raise ValueError(
            f"Override would cost ${est_cost:.4f}, exceeds ceiling "
            f"${plan['max_cost_usd']:.2f}"
        )
    est_cost = round(est_cost, 6)

    # Safety: check balance (the ledger is charged after provisioning, below).
    balance = await provider.get_balance()
    if balance <= 0:
        raise RuntimeError(
            "Account balance is $0. Top up before launching. "
            "Use the provider payment link."
        )
    if est_cost > balance:
        raise RuntimeError(
            f"Estimated cost ${est_cost:.4f} exceeds current balance ${balance:.2f}"
        )

    # Another launch/kill may have run during the awaits above.
    if plan["state"] != "planned":
        raise ValueError(
            f"Plan '{plan_id}' is in state '{plan['state']}' and cannot be launched"
        )

    plan.update(
        chip=new_chip,
        nodes=new_nodes,
        est_cost_usd=est_cost,
        est_minutes=round(est_minutes, 4),
        state="provisioning",  # blocks a concurrent second launch of this plan
        shards=[],
        results=[],
        combinations_evaluated=0,
        best_so_far=None,
        started_at=None,
        finished_at=None,
        killed_at=None,
    )

    await ctx.info(
        f"Launching plan {plan_id}: {plan['total']:,} combinations on "
        f"{plan['nodes']}×{plan['chip']} (est. ${plan['est_cost_usd']:.4f})"
    )

    # Provision, then charge. Any failure (or cancellation) stops every node
    # created so far and leaves the plan launchable again.
    node_names: list[str] = []
    try:
        for i in range(plan["nodes"]):
            name = f"{plan_id[:8]}-node-{i}"
            await provider.create_instance(name, plan["chip"])
            node_names.append(name)
            plan["shards"].append(
                {"node": name, "index": i, "status": "running", "scored": 0}
            )
        if plan["state"] != "provisioning":  # brute_genius_kill landed meanwhile
            raise RuntimeError(f"Plan '{plan_id}' was killed during provisioning")
        remaining = await provider.charge(plan["est_cost_usd"])
    except BaseException as exc:
        _log(f"launch {plan_id} failed while provisioning: {exc!r}; stopping nodes")
        await _stop_nodes(node_names)
        plan["shards"] = []
        if plan["state"] != "killed":
            plan["state"] = "planned"
        raise
    await ctx.info(
        f"Charged ${plan['est_cost_usd']:.4f}; balance now ${remaining:.2f}"
    )

    plan["state"] = "running"
    plan["started_at"] = time.time()

    fn = objectives.get(plan["objective"])
    device = xpu.pick_device(preferred=xpu.CHIP_DEVICE_HINTS.get(plan["chip"], "cpu"))
    plan["device_used"] = device
    await ctx.info(f"Scoring device for this run: {device}")

    shards = enumerator.shard(plan["space"], plan["nodes"])

    try:
        for i, shard_iter in enumerate(shards):
            if plan["state"] == "killed":
                await ctx.info(f"Plan {plan_id} killed before shard {i + 1}")
                break

            await ctx.report_progress(
                progress=i,
                total=plan["nodes"],
                message=f"Dispatching shard {i + 1}/{plan['nodes']}",
            )
            await ctx.info(f"Shard {i} → node {node_names[i]} on {device}")

            try:
                # Worker thread: the event loop stays free to serve status/kill.
                scored = await asyncio.to_thread(
                    xpu.batch_score, shard_iter, fn, device, plan=plan
                )
            except RuntimeError:
                if plan["state"] != "killed":
                    raise  # a real scoring error, not the kill signal
                await ctx.info(
                    f"Plan {plan_id} killed mid-shard {i + 1}; "
                    f"keeping {plan['combinations_evaluated']} partial results"
                )
                plan["shards"][i]["status"] = "stopped"
                break

            # Publish this shard's results immediately so status/results/kill
            # see live partials. Keep the list sorted best-first at all times.
            plan["results"].extend(scored)
            plan["results"].sort(key=lambda x: x[1], reverse=True)
            plan["shards"][i]["status"] = "done"
            plan["shards"][i]["scored"] = len(scored)
            plan["combinations_evaluated"] += len(scored)
            if scored:
                local_best = max(scored, key=lambda x: x[1])
                best = plan["best_so_far"]
                if best is None or local_best[1] > best["score"]:
                    plan["best_so_far"] = {"combo": local_best[0], "score": local_best[1]}

            await ctx.report_progress(
                progress=i + 1,
                total=plan["nodes"],
                message=f"Shard {i + 1}/{plan['nodes']} complete ({len(scored)} scored)",
            )
    except BaseException as exc:
        _log(f"launch {plan_id} failed while scoring: {exc!r}; stopping nodes")
        if plan["state"] != "killed":
            plan["state"] = "failed"
        plan["finished_at"] = time.time()
        await _stop_nodes(node_names)
        raise

    if plan["state"] != "killed":
        plan["state"] = "done"
    plan["finished_at"] = time.time()

    # Stop billing on the nodes
    await _stop_nodes(node_names)

    await ctx.info(
        f"Plan {plan_id} finished with state={plan['state']}. "
        f"Evaluated {plan['combinations_evaluated']:,} combinations on {device}."
    )

    return {
        "plan_id": plan_id,
        "state": plan["state"],
        "combinations_evaluated": plan["combinations_evaluated"],
        "device": device,
        "chip": plan["chip"],
        "nodes": plan["nodes"],
        "estimated_cost_usd": plan["est_cost_usd"],
        "balance_usd": remaining,
        "best_so_far": plan["best_so_far"],
    }


@mcp.tool()
async def brute_genius_status(plan_id: str) -> dict[str, Any]:
    """Return live progress, shards done, combinations evaluated, best-so-far."""
    plan = _plans.get(plan_id)
    if plan is None:
        raise ValueError(f"Unknown plan_id '{plan_id}'")

    shards_done = sum(1 for s in plan["shards"] if s.get("status") == "done")
    return {
        "plan_id": plan_id,
        "state": plan["state"],
        "total_combinations": plan["total"],
        "combinations_evaluated": plan["combinations_evaluated"],
        "shards_total": plan["nodes"],
        "shards_done": shards_done,
        "best_so_far": plan["best_so_far"],
        "estimated_cost_usd": plan["est_cost_usd"],
        "device_used": plan.get("device_used"),
        "started_at": plan.get("started_at"),
        "finished_at": plan.get("finished_at"),
        "killed_at": plan.get("killed_at"),
    }


@mcp.tool()
async def brute_genius_results(
    plan_id: str,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Return the top_k highest-scoring combinations (descending score)."""
    plan = _plans.get(plan_id)
    if plan is None:
        raise ValueError(f"Unknown plan_id '{plan_id}'")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    results = plan["results"][:top_k]
    return [{"combo": c, "score": s} for c, s in results]


@mcp.tool()
async def brute_genius_kill(plan_id: str) -> dict[str, Any]:
    """Stop every node and shard belonging to a plan. Ends billing.

    Results from shards that already finished are preserved (a shard that is
    interrupted mid-way is discarded). Already-charged cost is not refunded.
    Reachable while scoring runs, because scoring happens off the event loop.
    """
    plan = _plans.get(plan_id)
    if plan is None:
        raise ValueError(f"Unknown plan_id '{plan_id}'")

    # Mark killed first so any in-flight launch / batch_score loop will abort
    plan["state"] = "killed"
    plan["killed_at"] = time.time()
    plan["finished_at"] = plan["killed_at"]

    # Stop every node associated with this plan
    stopped = 0
    for shard_meta in plan["shards"]:
        name = shard_meta.get("node")
        if name:
            try:
                await provider.stop_instance(name)
                shard_meta["status"] = "stopped"
                stopped += 1
            except Exception as exc:
                _log(f"kill {plan_id}: failed to stop node {name}: {exc!r}")

    return {
        "plan_id": plan_id,
        "state": "killed",
        "killed_at": plan["killed_at"],
        "nodes_stopped": stopped,
        "partial_results": len(plan["results"]),
        "combinations_evaluated": plan["combinations_evaluated"],
    }


@mcp.tool()
async def xpu_probe(chip: str = "h100") -> dict[str, Any]:
    """Detect and report accelerators on the machine running this server.

    Args:
        chip: Preferred chip name (informational; detection is exhaustive).
    """
    report = xpu.detect_accelerators()
    report["requested_chip"] = chip.lower()
    report["provider_rates"] = dict(provider.RATES)
    report["balance_usd"] = await provider.get_balance()
    return report


# ─── RESOURCES ─────────────────────────────────────────────────────────────

@mcp.resource("brutegenius://plans")
def list_plans() -> dict[str, Any]:
    """List of all plans and their current states."""
    return {
        pid: {
            "state": p["state"],
            "total": p["total"],
            "objective": p["objective"],
            "chip": p["chip"],
            "nodes": p["nodes"],
            "est_cost_usd": p["est_cost_usd"],
            "combinations_evaluated": p["combinations_evaluated"],
        }
        for pid, p in _plans.items()
    }


@mcp.resource("brutegenius://plans/{plan_id}/combinations")
def plan_combinations(plan_id: str) -> dict[str, Any]:
    """Paginated view of the combination set for a plan.

    Returns the first 1 000 combinations together with total count.
    Full materialisation of multi-million spaces is intentionally avoided.
    """
    plan = _plans.get(plan_id)
    if plan is None:
        raise ValueError(f"Unknown plan_id '{plan_id}'")

    # Stream only the first page
    page: list[dict[str, Any]] = []
    for i, combo in enumerate(enumerator.cartesian(plan["space"])):
        if i >= 1000:
            break
        page.append(combo)

    return {
        "plan_id": plan_id,
        "total": plan["total"],
        "page_size": 1000,
        "returned": len(page),
        "combinations": page,
        "note": "First 1 000 combinations only. Use sharding for full evaluation.",
    }


@mcp.resource("brutegenius://hardware")
def hardware_inventory() -> dict[str, Any]:
    """Live inventory of accelerators visible to this process."""
    return xpu.detect_accelerators()


@mcp.resource("brutegenius://pricing")
async def pricing() -> dict[str, Any]:
    """Current burn rate per chip type (USD per hour) and account balance."""
    return {
        "rates_usd_per_hour": dict(provider.RATES),
        "balance_usd": await provider.get_balance(),
        "currency": "USD",
    }


# ─── PROMPTS ───────────────────────────────────────────────────────────────

@mcp.prompt()
def define_search() -> str:
    """Walk the user through specifying a search space, objective, and cost ceiling."""
    available = ", ".join(objectives.list_objectives())
    return f"""You are helping the user define an exhaustive combinatorial search for Brute Genius.

Collect the following information step by step:

1. **Search-space axes**  
   Ask the user for each discrete axis name and the list of values it can take.  
   Example: {{"lr": [1e-4, 3e-4, 1e-3], "batch": [16, 32, 64], "dropout": [0.0, 0.1, 0.2]}}

2. **Objective function**  
   The objective must be one of the registered names: {available}.  
   Explain briefly what each does if the user is unsure.

3. **Cost ceiling**  
   Ask for a hard maximum spend in USD (max_cost_usd).  
   Brute Genius will refuse to create a plan that exceeds this number.

4. **Optional hardware**  
   Chip type (default h100) and node count (default 1, max 64).

Once you have the answers, call the tool `brute_genius_plan` with the collected arguments and present the returned cost estimate and plan_id to the user. Do NOT call `brute_genius_launch` until the user explicitly confirms they accept the cost.
"""


@mcp.prompt()
def interpret_results() -> str:
    """Ask the LLM to explain the top results and any visible patterns."""
    return """The user has finished a Brute Genius run. You will be given the top_k results (combinations + scores).

Your job:
1. Identify the single best combination and state its score clearly.
2. Look for patterns across the top results (e.g. "every winner used lr=3e-4", "batch size 32 dominates").
3. Call out any surprising losers or plateaus.
4. Suggest one or two follow-up search spaces that would refine the current winners (narrower ranges around the best values).
5. Remind the user of the total combinations evaluated and the final state of the plan.

Be concise and quantitative. Do not invent results that were not supplied.
"""


# ─── ENTRY POINT ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Default transport is stdio (local clients). For remote/HTTP:
    #   mcp.run(transport="http", host="0.0.0.0", port=8000)
    mcp.run()
