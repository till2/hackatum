from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Order:
    order_id: int
    instrument_id: str
    price: float
    volume: int
    side: str  # 'bid' or 'ask'
    order_type: str  # e.g., 'limit', 'ioc'
    user_id: str # who placed the order

@dataclass
class Trade:
    trade_id: float
    instrument_id: str
    price: float
    volume: int
    aggressor: str # 'buyer' or 'seller'
    buyer_id: str
    seller_id: str

@dataclass
class PriceBook:
    bid_prices: List[float] = field(default_factory=list)
    ask_prices: List[float] = field(default_factory=list)


@dataclass
class SocialMediaFeed:
    feed_id: int
    content: str
    timestamp: float 

@dataclass
class Instrument:
    instrument_id: str
    price: float
    volume: int
