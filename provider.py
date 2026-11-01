"""Abstract GPUProvider interface.

Swap MockProvider for a real cloud provider (RunPod, Vultr, AWS, etc.)
without touching server.py. All money-related operations go through this
interface so cost ceilings can be enforced before any node is created.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class GPUProvider(ABC):
    """Contract every fleet backend must satisfy."""

    # Hourly rates in USD. Subclasses may override or extend.
    RATES: dict[str, float] = {
        "h100": 0.084,
        "a100": 0.050,
        "l40s": 0.040,
        "cpu": 0.003,
    }

    @abstractmethod
    async def create_instance(self, name: str, chip: str) -> dict[str, Any]:
        """Provision a new accelerator node. Returns node metadata."""
        ...

    @abstractmethod
    async def stop_instance(self, name: str) -> bool:
        """Stop billing for the named node. Returns True on success."""
        ...

    @abstractmethod
    async def run_command(self, name: str, command: str) -> str:
        """Execute a command on the node and return stdout."""
        ...

    @abstractmethod
    async def get_balance(self) -> float:
        """Current account balance in USD."""
        ...

    @abstractmethod
    async def charge(self, amount_usd: float) -> float:
        """Deduct ``amount_usd`` from the balance and return the new balance.

        Must raise (never silently succeed) if the amount cannot be charged:
        RuntimeError for insufficient funds, NotImplementedError if the backend
        has no way to bill from here.
        """
        ...

    @abstractmethod
    async def get_payment_link(self, amount: int) -> str:
        """Return a URL the user can open to top up `amount` USD."""
        ...

    def hourly_rate(self, chip: str) -> float:
        """Look up the hourly burn rate for a chip type."""
        return self.RATES.get(chip.lower(), self.RATES.get("h100", 0.084))
