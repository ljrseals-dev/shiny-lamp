"""In-memory fleet for testing and local development.

Replace with a real provider when you are ready to spend real money.
All state lives in process memory and disappears when the server exits.
"""
from __future__ import annotations

from typing import Any

from provider import GPUProvider


class MockProvider(GPUProvider):
    """Simulated GPU fleet with a fixed starting balance."""

    RATES = {
        "h100": 0.084,
        "a100": 0.050,
        "l40s": 0.040,
        "cpu": 0.003,
    }

    def __init__(self, balance: float = 25.0) -> None:
        self.balance = float(balance)
        self._nodes: dict[str, dict[str, Any]] = {}
        self._stopped: set[str] = set()

    async def create_instance(self, name: str, chip: str) -> dict[str, Any]:
        if name in self._nodes and name not in self._stopped:
            raise ValueError(f"Node '{name}' already exists and is running")
        chip = chip.lower()
        if chip not in self.RATES:
            raise ValueError(f"Unknown chip '{chip}'. Known: {list(self.RATES)}")
        self._nodes[name] = {
            "name": name,
            "chip": chip,
            "status": "running",
            "rate_usd_per_hour": self.RATES[chip],
        }
        self._stopped.discard(name)
        return dict(self._nodes[name])

    async def stop_instance(self, name: str) -> bool:
        if name not in self._nodes:
            raise ValueError(f"Node '{name}' not found")
        self._nodes[name]["status"] = "stopped"
        self._stopped.add(name)
        return True

    async def run_command(self, name: str, command: str) -> str:
        if name not in self._nodes:
            raise ValueError(f"Node '{name}' not found")
        if self._nodes[name]["status"] != "running":
            raise RuntimeError(f"Node '{name}' is not running")
        cmd = command.strip()
        if cmd == "nvidia-smi":
            chip = self._nodes[name]["chip"]
            return f"NVIDIA {chip.upper()} 80GB HBM3 (simulated)"
        return f"[simulated on {name}] {cmd}"

    async def get_balance(self) -> float:
        return self.balance

    async def charge(self, amount_usd: float) -> float:
        if amount_usd < 0:
            raise ValueError("amount_usd must be >= 0")
        if amount_usd > self.balance:
            raise RuntimeError(
                f"Insufficient balance: need ${amount_usd:.4f}, have ${self.balance:.4f}"
            )
        self.balance -= amount_usd
        return self.balance

    async def get_payment_link(self, amount: int) -> str:
        return f"https://pay.example.com/topup?amount={amount}"

    # ── helpers used by the server ──────────────────────────────────────

    def list_nodes(self) -> list[dict[str, Any]]:
        return [dict(n) for n in self._nodes.values()]

    def stop_all(self) -> int:
        """Stop every running node. Returns the number stopped."""
        count = 0
        for name, meta in self._nodes.items():
            if meta["status"] == "running":
                meta["status"] = "stopped"
                self._stopped.add(name)
                count += 1
        return count
