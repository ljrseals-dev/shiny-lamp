"""Accelerator detection and batch scoring.

Detects available accelerators. Scoring currently runs on the host CPU; the
device is recorded for logging and future dispatch (see the README Roadmap,
items C2 and C3).
"""
from __future__ import annotations

import subprocess
from typing import Any, Callable, Iterator

# Preferred device family for each chip name the providers know about. Used by
# brute_genius_launch to choose which detected device to record for a run.
CHIP_DEVICE_HINTS: dict[str, str] = {
    "h100": "cuda",
    "a100": "cuda",
    "l40s": "cuda",
    "4090": "cuda",
    "cpu": "cpu",
}


def detect_accelerators() -> dict[str, Any]:
    """Query nvidia-smi, torch.xpu, and torch.backends.mps.

    Returns a unified hardware report:
      {
        "cuda": [{"name": "...", "memory_mib": int}, ...],
        "xpu":  [{"name": "..."}, ...],
        "mps":  bool,
        "cpu":  True,
        "preferred": "cuda" | "xpu" | "mps" | "cpu"
      }
    """
    report: dict[str, Any] = {
        "cuda": [],
        "xpu": [],
        "mps": False,
        "cpu": True,
        "preferred": "cpu",
    }

    # NVIDIA GPUs via nvidia-smi
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            for line in out.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    try:
                        mem = int(float(parts[1]))
                    except ValueError:
                        mem = 0
                    report["cuda"].append({"name": parts[0], "memory_mib": mem})
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # Intel XPU and Apple MPS via torch (optional dependency)
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            for i in range(torch.xpu.device_count()):
                name = torch.xpu.get_device_name(i)
                report["xpu"].append({"name": name})
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            report["mps"] = True
    except ImportError:
        pass

    # Preferred device
    if report["cuda"]:
        report["preferred"] = "cuda"
    elif report["xpu"]:
        report["preferred"] = "xpu"
    elif report["mps"]:
        report["preferred"] = "mps"
    else:
        report["preferred"] = "cpu"

    return report


def pick_device(preferred: str = "cuda") -> str:
    """Return the best available device string given a preference."""
    rep = detect_accelerators()
    if preferred == "cuda" and rep["cuda"]:
        return "cuda"
    if preferred == "xpu" and rep["xpu"]:
        return "xpu"
    if preferred == "mps" and rep["mps"]:
        return "mps"
    if preferred == "cpu":
        return "cpu"
    # Fall back to whatever is actually present
    return rep["preferred"]


def batch_score(
    shard: Iterator[dict[str, Any]],
    fn: Callable[[dict[str, Any]], float],
    device: str,
    batch_size: int = 256,
    plan: dict[str, Any] | None = None,
) -> list[tuple[dict[str, Any], float]]:
    """Score a shard in batches, aborting if plan['state'] == 'killed'.

    `fn` is a plain callable from the objectives registry — never a string.
    The device argument is recorded for logging only: this implementation runs
    the pure-Python objective on the host CPU, in whatever thread calls it.
    Vectorised/torch scoring is Roadmap C2; shipping the shard + objective name
    to remote nodes is Roadmap C3.

    Checks plan state per combination (cheap path) and again before every
    batch evaluation (expensive path), so the worst case after kill() is
    one in-flight combination, not one in-flight batch.
    """
    results: list[tuple[dict[str, Any], float]] = []
    batch: list[dict[str, Any]] = []

    def _check() -> None:
        if plan is not None and plan.get("state") == "killed":
            raise RuntimeError("Plan killed")

    for combo in shard:
        _check()  # cheap, per-combination
        batch.append(combo)
        if len(batch) >= batch_size:
            _check()  # also before the expensive call
            results.extend((c, fn(c)) for c in batch)
            batch = []

    if batch:
        _check()
        results.extend((c, fn(c)) for c in batch)

    return results
