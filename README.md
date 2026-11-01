# Brute Genius

Exhaustive combinatorial search as an MCP server. Given a discrete search space, Brute Genius enumerates the full Cartesian product, shards it, scores every combination against a named objective, and returns a ranked list. It does not sample or prune; it tries everything.

**Current build, stated plainly:** shards are evaluated *sequentially, in-process, on the host* (in a worker thread so `status` and `kill` stay responsive). Nodes are provisioned and billed through a `GPUProvider` seam, but they do not execute the work yet: remote execution is on the [Roadmap](#roadmap). The only provider wired into the server is `MockProvider`, whose money is simulated, so nothing is spent today. `RunPodProvider` exists but cannot run launches yet (`run_command` and `charge` are not implemented).

Always call `brute_genius_plan` first and respect the cost ceiling it returns. `brute_genius_kill` is the emergency stop.

## Connect to Claude Desktop

1. Install dependencies:

```bash
cd brute-genius
pip install -e ".[test]"
```

2. Add the server to Claude Desktop’s MCP configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "brute-genius": {
      "command": "python",
      "args": ["/absolute/path/to/brute-genius/server.py"]
    }
  }
}
```

3. Restart Claude Desktop. The tools and resources below will appear, along with the prompts.

For other clients (Cursor, Copilot, etc.) point the MCP host at the same `python /absolute/path/to/server.py` stdio command. For remote/HTTP use:

```python
mcp.run(transport="http", host="0.0.0.0", port=8000)
```

## Tools and resources

| Name | Kind | What it does |
| --- | --- | --- |
| `brute_genius_plan` | tool | Counts the space, estimates cost, stores the ceiling. Executes nothing. |
| `brute_genius_launch` | tool | Re-checks the ceiling and balance, provisions nodes, charges the estimate, scores shards sequentially on the host. |
| `brute_genius_status` | tool | Live state, shards done, combinations evaluated, best-so-far. |
| `brute_genius_results` | tool | Top-k combinations, best first (partial while a run is live). |
| `brute_genius_kill` | tool | Stops every node for a plan and aborts scoring. |
| `xpu_probe` | tool | Reports detected accelerators, provider rates and balance. |
| `brutegenius://plans` | resource | All plans and their states. |
| `brutegenius://plans/{plan_id}/combinations` | resource | First 1 000 combinations of a plan. |
| `brutegenius://hardware` | resource | Accelerators visible to this process. |
| `brutegenius://pricing` | resource | Rates per chip and the account balance, read through the provider's async `get_balance()`. |

## Quick test

```bash
pytest test_server.py -v
npx @modelcontextprotocol/inspector python server.py
```

## Safety guarantees

Each of these is covered by a test in `test_server.py`.

- `plan()` refuses any estimate above `max_cost_usd` and stores that ceiling in the plan.
- `launch()` recomputes the cost after any `chip`/`nodes` override and refuses anything above the plan's ceiling, before any node is created. A refused override leaves the plan unchanged.
- `launch()` refuses when the estimate exceeds the provider balance, and otherwise **charges the estimate against the balance before scoring** (`provider.charge`; insufficient funds raise). There is no refund on early `kill` yet (Roadmap).
- If provisioning fails partway, or scoring raises, `launch()` stops every node it created. A provisioning failure leaves the plan `planned` and relaunchable with clean counters; a scoring failure marks it `failed`.
- A `killed` or `failed` plan cannot be relaunched; create a new plan. A `done` plan cannot be launched again.
- `kill()` is reachable while scoring runs, stops every node associated with the plan_id, and reports the results of all shards that finished. A shard interrupted mid-way is discarded, not partially kept.
- Hard caps: 64 nodes, 10⁹ combinations, at least one axis in the space.
- Objectives are registry-only; caller code is never executed.

## Roadmap

**Near-term (this pass)**

- [x] Index-based sharding: each shard decodes only its own indices (A1)
- [x] Cost ceiling re-enforced at launch (A2)
- [x] Ledger-backed balance with `charge()` (A3)
- [x] Coherent relaunch semantics: refuse killed/failed/done; reset a rolled-back `planned` plan (A4)
- [x] Incremental results so status/results/kill see live partials (A5)
- [x] Provisioning rollback (A6)

**Next**

- [ ] **C1. True parallelism.** Dispatch shards concurrently with `asyncio.gather` over `xpu.batch_score` running in `run_in_executor`, or move scoring into worker processes. Add a `concurrency` parameter (default 1, capped at `MAX_NODES`).
- [ ] **C2. Vectorized objectives.** Provide a torch-backed variant of `sphere`/`rastrigin`/`rosenbrock` that batches a shard into a tensor and returns scores; keep the pure-Python path as the fallback and as the reference for tests. `device` then actually influences execution.
- [ ] **C3. Remote execution.** Implement `RunPodProvider.run_command` via SSH/paramiko, or bake a scoring entrypoint image; add `GPUProvider.run_shard(...)` so shards ship to nodes instead of being scored locally.

**Later**

- [ ] Plan TTL / eviction for `_plans` (bounded LRU, e.g. 256 entries, 24 h TTL).
- [ ] Refund the unused estimate on early kill (needs a per-second billing surface).
- [ ] Resume-from-checkpoint per shard using the index-based decoder (persist `k_last` per shard); this would also keep the partial work of a shard that is killed mid-way.
- [ ] Structured `_log` JSON lines with `plan_id`, `shard_idx`, `event`.
- [ ] Optional webhook / `ctx.set_status` for plan lifecycle events.
