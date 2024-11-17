import threading
from collections import defaultdict
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass(order=True)
class Order:
    sort_index: float = field(init=False, repr=False)
    price: float
    timestamp: float
    order_id: str
    instrument: str
    side: str  # 'bid' or 'ask'
    volume: int
    order_type: str  # 'limit' or 'ioc'

    def __post_init__(self):
        # For sorting: bids sorted descending, asks ascending
        if self.side == 'bid':
            # Negative price for max-heap behavior in sorted list
            self.sort_index = -self.price
        else:
            self.sort_index = self.price

@dataclass
class PriceVolume:
    price: float
    volume: int

@dataclass
class PriceBook:
    bids: List[PriceVolume] = field(default_factory=list)
    asks: List[PriceVolume] = field(default_factory=list)

class Exchange:
    """
    A simulated exchange that handles order matching and trade execution.

    The Exchange maintains order books for multiple instruments, tracks outstanding orders,
    manages positions and cash balances, and provides thread-safe access to trading operations.

    Attributes:
        order_books (Dict[str, PriceBook]): Maps instrument IDs to their price books
        outstanding_orders (Dict[str, Dict[str, Order]]): Maps instruments to their outstanding orders
        positions (Dict[str, int]): Current position sizes for each instrument
        cash (float): Available cash balance
        lock (threading.Lock): Thread lock for synchronization
    """
    def __init__(self):
        self.order_books: Dict[str, PriceBook] = defaultdict(PriceBook)
        self.outstanding_orders: Dict[str, Dict[str, Order]] = defaultdict(dict)
        self.positions: Dict[str, float] = defaultdict(float)
        self.cash: float = 0.0
        self.lock = threading.Lock()

    def connect(self):
        """
        Simulate connecting to the exchange.
        """
        # In a real implementation, connection setup would occur here.
        # For simulation, we'll initialize empty order books.
        print("Connected to the Exchange.")
        return True

    def get_last_price_book(self, instrument: str) -> PriceBook:
        """
        Retrieve the latest price book for the specified instrument.
        """
        with self.lock:
            return self.order_books[instrument]

    def insert_order(self, instrument: str, price: float, side: str, volume: int, order_type: str) -> Dict:
        """
        Insert a new order into the exchange.
        """
        with self.lock:
            order_id = str(uuid.uuid4())
            timestamp = time.time()
            order = Order(price=price, timestamp=timestamp, order_id=order_id,
                          instrument=instrument, side=side, volume=volume, order_type=order_type)
            
            trades = self.match_order(order)
            
            # If IOC and not fully filled, cancel the remaining
            if order.order_type == 'ioc' and order.volume > 0:
                # Remaining volume is canceled
                pass  # No action needed as it's already not in the order book
            elif order.volume > 0:
                # Add to order book
                self.add_order_to_book(order)
                self.outstanding_orders[instrument][order_id] = order

            return {
                'status': 'success',
                'order_id': order_id,
                'trades': trades
            }

    def match_order(self, incoming_order: Order) -> List[Dict]:
        """
        Match the incoming order against existing orders in the order book.
        """
        trades = []
        book = self.order_books[incoming_order.instrument]
        opposite_side = 'ask' if incoming_order.side == 'bid' else 'bid'
        orders = sorted(
            [o for o in self.outstanding_orders[incoming_order.instrument].values() if o.side == opposite_side],
            key=lambda x: (x.price, x.timestamp)
        ) if incoming_order.side == 'bid' else sorted(
            [o for o in self.outstanding_orders[incoming_order.instrument].values() if o.side == opposite_side],
            key=lambda x: (-x.price, x.timestamp)
        )

        i = 0
        while incoming_order.volume > 0 and i < len(orders):
            current_order = orders[i]
            price_match = (incoming_order.side == 'bid' and incoming_order.price >= current_order.price) or \
                          (incoming_order.side == 'ask' and incoming_order.price <= current_order.price)
            if not price_match:
                break

            traded_volume = min(incoming_order.volume, current_order.volume)
            trade_price = current_order.price

            # Update volumes
            incoming_order.volume -= traded_volume
            current_order.volume -= traded_volume

            # Update positions
            if incoming_order.side == 'bid':
                self.positions[incoming_order.instrument] += traded_volume
                self.positions[self.get_counter_instrument(instrument=incoming_order.instrument)] -= traded_volume
                self.cash -= traded_volume * trade_price
                self.cash += traded_volume * trade_price
            else:
                self.positions[incoming_order.instrument] -= traded_volume
                self.positions[self.get_counter_instrument(instrument=incoming_order.instrument)] += traded_volume
                self.cash += traded_volume * trade_price
                self.cash -= traded_volume * trade_price

            trades.append({
                'buy_order_id': incoming_order.order_id if incoming_order.side == 'bid' else current_order.order_id,
                'sell_order_id': current_order.order_id if incoming_order.side == 'bid' else incoming_order.order_id,
                'price': trade_price,
                'volume': traded_volume
            })

            if current_order.volume == 0:
                # Remove the order from the order book
                del self.outstanding_orders[incoming_order.instrument][current_order.order_id]
            else:
                # Update the order with remaining volume
                self.outstanding_orders[incoming_order.instrument][current_order.order_id] = current_order

            i += 1

        return trades

    def add_order_to_book(self, order: Order):
        """
        Add the order to the appropriate side of the order book.
        """
        if order.side == 'bid':
            self.order_books[order.instrument].bids.append(PriceVolume(price=order.price, volume=order.volume))
            # Sort bids descending
            self.order_books[order.instrument].bids.sort(key=lambda x: -x.price)
        else:
            self.order_books[order.instrument].asks.append(PriceVolume(price=order.price, volume=order.volume))
            # Sort asks ascending
            self.order_books[order.instrument].asks.sort(key=lambda x: x.price)

    def get_outstanding_orders(self, instrument: str) -> Dict[str, Order]:
        """
        Retrieve all outstanding orders for the specified instrument.
        """
        with self.lock:
            return self.outstanding_orders[instrument]

    def delete_order(self, instrument: str, order_id: str) -> Dict:
        """
        Delete a specific order from the order book.
        """
        with self.lock:
            if order_id in self.outstanding_orders[instrument]:
                del self.outstanding_orders[instrument][order_id]
                return {'status': 'success', 'order_id': order_id}
            else:
                return {'status': 'failure', 'reason': 'Order ID not found'}

    def get_positions(self) -> Dict[str, float]:
        """
        Get current positions for all instruments.
        """
        with self.lock:
            return dict(self.positions)

    def get_positions_and_cash(self) -> Dict[str, float]:
        """
        Get current positions along with cash balance.
        """
        with self.lock:
            positions = dict(self.positions)
            positions['cash'] = self.cash
            return positions

    def get_counter_instrument(self, instrument: str) -> str:
        """
        Assuming a dual-listing, return the counterpart instrument.
        """
        return 'PHILIPS_A' if instrument == 'PHILIPS_B' else 'PHILIPS_B'
