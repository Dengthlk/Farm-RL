from __future__ import annotations

from dataclasses import dataclass
from kaggriculture.actions import MarketKind


SELL_SET_KINDS = tuple(kind for kind in MarketKind if kind.name.startswith("SELL_"))


MARKET_SET_KINDS = (
    *SELL_SET_KINDS,
    MarketKind.HIRE,
    MarketKind.BUY_LAND,
    *(kind for kind in MarketKind if kind.name.startswith("BUY_SEED_")),
    *(kind for kind in MarketKind if kind.name.startswith("BUY_ANIMAL_")),
    *(kind for kind in MarketKind if kind.name.startswith("BUY_PRODUCT_")),
)


N_MARKET_SET_KINDS = len(MARKET_SET_KINDS)


@dataclass(frozen=True)
class MarketSetOrder:
    """The Stage-0b-selected convention, specified explicitly by every caller."""

    sell_order: str = "fixed"
    hire_last: bool = False

    def __post_init__(self) -> None:
        if self.sell_order not in {"fixed", "impact"}:
            raise ValueError(f"unknown sell order {self.sell_order!r}")

    @property
    def decision_kinds(self) -> tuple[MarketKind, ...]:
        if not self.hire_last:
            return MARKET_SET_KINDS
        return (*SELL_SET_KINDS, *MARKET_SET_KINDS[len(SELL_SET_KINDS) + 1 :], MarketKind.HIRE)
