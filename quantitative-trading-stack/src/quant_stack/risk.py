"""Pure-Python risk controls (no Nautilus imports) so they are unit-testable in isolation."""

from __future__ import annotations

import math


class DrawdownMonitor:
    """Tracks the equity high-water mark and trips when drawdown reaches the limit."""

    def __init__(self, max_drawdown: float) -> None:
        if not 0 < max_drawdown < 1:
            raise ValueError("max_drawdown must be in (0, 1)")
        self.max_drawdown = max_drawdown
        self.peak = 0.0
        self.tripped = False
        self.drawdown = 0.0

    def update(self, equity: float) -> bool:
        """Feed the latest equity. Returns True once the kill-switch has tripped (latched)."""
        if not math.isfinite(equity) or equity <= 0:
            self.tripped = True
            return True
        self.peak = max(self.peak, equity)
        self.drawdown = (self.peak - equity) / self.peak
        if self.drawdown >= self.max_drawdown:
            self.tripped = True
        return self.tripped


def slice_quantity(quantity: int, price: float, max_order_value: float) -> list[int]:
    """Split an order into child orders that each respect ``max_order_value``."""
    if quantity <= 0:
        return []
    if price <= 0:
        raise ValueError("price must be positive")
    max_qty = int(max_order_value // price)
    if max_qty < 1:
        raise ValueError(f"max_order_value {max_order_value} is below one share at {price}")
    full, rest = divmod(quantity, max_qty)
    return [max_qty] * full + ([rest] if rest else [])
