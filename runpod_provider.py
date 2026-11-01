"""RunPod-backed provider. Swap this in for MockProvider when you're ready to spend.

Requires:
  pip install runpod
  export RUNPOD_API_KEY=...

Caveats:
  - run_command is intentionally stubbed: the RunPod Python SDK manages pods,
    not shells inside them. Bake the scoring workload into a Docker image and
    set it as the pod entrypoint, or wire SSH/paramiko yourself.
  - get_balance uses runpod.get_user(); if your SDK version differs, fall back
    to the REST billing endpoint.
  - charge is intentionally not implemented: RunPod bills per pod-second on the
    account and exposes no "deduct this amount" call. brute_genius_launch
    charges before scoring, so launching on this provider currently fails
    *after* provisioning and rolls the pods back (stop_pod). This provider is
    not wired into server.py; see the README Roadmap.
  - Cost ceilings are enforced by brute_genius_plan and re-enforced (after any
    chip/nodes override) by brute_genius_launch before any pod is created.
"""
from __future__ import annotations

import os
from typing import Any

from provider import GPUProvider

try:
    import runpod
except ImportError:  # pragma: no cover
    runpod = None  # type: ignore[assignment]


class RunPodProvider(GPUProvider):
    """GPUProvider implementation backed by the official RunPod Python SDK."""

    # Short names → RunPod gpu_type_id values.
    CHIP_MAP: dict[str, str | None] = {
        "h100": "NVIDIA H100 80GB HBM3",
        "a100": "NVIDIA A100 80GB PCIe",
        "l40s": "NVIDIA L40S",
        "4090": "NVIDIA GeForce RTX 4090",
        "cpu": None,  # CPU pods omit gpu_type_id
    }

    RATES = {
        "h100": 0.084,
        "a100": 0.050,
        "l40s": 0.040,
        "4090": 0.030,
        "cpu": 0.003,
    }

    def __init__(self, api_key: str | None = None) -> None:
        if runpod is None:
            raise ImportError(
                "runpod package is required for RunPodProvider. "
                "Install with: pip install runpod"
            )
        key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not key:
            raise ValueError(
                "RUNPOD_API_KEY must be set in the environment or passed to "
                "RunPodProvider(api_key=...)"
            )
        runpod.api_key = key
        self._pods: dict[str, str] = {}  # our name → RunPod pod id

    async def create_instance(self, name: str, chip: str) -> dict[str, Any]:
        if name in self._pods:
            raise ValueError(f"Node '{name}' already exists")

        chip_key = chip.lower()
        if chip_key not in self.CHIP_MAP:
            raise ValueError(
                f"Unsupported chip '{chip}'. Use one of {list(self.CHIP_MAP)}"
            )
        gpu = self.CHIP_MAP[chip_key]

        # Point image_name at a scoring image you control. runpod/stack is a
        # generic base; replace with your own for real workloads.
        kwargs: dict[str, Any] = {
            "name": name,
            "image_name": os.environ.get("BRUTE_GENIUS_IMAGE", "runpod/stack"),
            "container_disk_in_gb": 20,
            "volume_in_gb": 0,
            "ports": "22/tcp",
        }
        if gpu is not None:
            kwargs["gpu_type_id"] = gpu

        pod = runpod.create_pod(**kwargs)
        if not pod or "id" not in pod:
            raise RuntimeError(f"RunPod create failed for '{name}': {pod}")

        self._pods[name] = pod["id"]
        return {
            "name": name,
            "chip": chip_key,
            "pod_id": pod["id"],
            "status": pod.get("desiredStatus", "running"),
        }

    async def stop_instance(self, name: str) -> bool:
        pod_id = self._pods.get(name)
        if not pod_id:
            raise ValueError(f"Node '{name}' not tracked")
        result = runpod.stop_pod(pod_id)
        if not result:
            raise RuntimeError(f"RunPod stop failed for '{name}' ({pod_id})")
        return True

    async def run_command(self, name: str, command: str) -> str:
        # The SDK has no remote-exec. Options:
        #   1. Bake scoring into the pod image entrypoint.
        #   2. Add paramiko SSH against the exposed 22/tcp port.
        raise NotImplementedError(
            "run_command needs SSH or an image entrypoint. "
            "Bake the scoring workload into the pod image, or implement "
            "paramiko against the exposed 22/tcp port."
        )

    async def get_balance(self) -> float:
        # SDK surface varies by version; try the common entry points.
        try:
            info = runpod.get_user()
        except AttributeError:
            try:
                info = runpod.api.account_info()  # type: ignore[attr-defined]
            except Exception as exc:
                raise RuntimeError(
                    f"Unable to read RunPod balance: {exc}"
                ) from exc
        return float(info.get("balance", 0.0) if isinstance(info, dict) else 0.0)

    async def charge(self, amount_usd: float) -> float:
        # Do not pretend: RunPod has no per-amount charge API.
        raise NotImplementedError(
            "RunPodProvider cannot deduct a fixed amount: RunPod bills per pod-second "
            "on the account. Implement charge() against a billing surface you control "
            "before using this provider with brute_genius_launch."
        )

    async def get_payment_link(self, amount: int) -> str:
        # RunPod has no per-amount payment-link API.
        return "https://www.runpod.io/console/user/billing"
