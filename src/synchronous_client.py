import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from common_types import Order, PriceBook, Trade
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Configure logging with colored formatter
class ColorFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages based on their level."""

    LEVEL_COLOR = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        color = self.LEVEL_COLOR.get(record.levelno, Fore.WHITE)
        record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)

# Set up logging with the custom formatter
handler = logging.StreamHandler()
formatter = ColorFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = [handler]

class Exchange:
    """
    A simulated exchange for handling orders, trades, and maintaining positions and cash.
    """

    def __init__(self):
        self.connected = False
        self.instruments = ["AAPL", "GOOG", "MSFT"]
        self.orders: Dict[str, Dict[int, Order]] = {instrument: {} for instrument in self.instruments}
        self.trade_history: Dict[str, List[Trade]] = {instrument: [] for instrument in self.instruments}
        self.trade_id_counter = 1
        self.positions: Dict[str, int] = {instrument: 0 for instrument in self.instruments}
        self.cash = 100000.0  # Starting with $100,000
        self.price_books: Dict[str, PriceBook] = {instrument: PriceBook() for instrument in self.instruments}
        self.next_order_id = 1
        logger.info("Exchange initialized.")

    def connect(self):
        """Connect to the exchange simulator."""
        if not self.connected:
            self.connected = True
            logger.info("Connected to Exchange Simulator.")
        else:
            logger.info("Already connected to Exchange Simulator.")

    def disconnect(self):
        """Disconnect from the exchange simulator."""
        if self.connected:
            self.connected = False
            logger.info("Disconnected from Exchange Simulator.")
        else:
            logger.info("Already disconnected from Exchange Simulator.")

    def get_instruments(self) -> List[str]:
        """Return a list of available instruments."""
        return self.instruments

    def insert_order(self, instrument_id: str, price: float, volume: int, side: str, order_type: str, user_id: str) -> Order:
        """
        Insert a new order into the order book.

        Args:
            instrument_id (str): The ID of the instrument.
            price (float): The price of the order.
            volume (int): The volume of the order.
            side (str): 'bid' for buy orders or 'ask' for sell orders.
            order_type (str): Type of the order, e.g., 'limit', 'ioc'.
            user_id (str): The ID of the user placing the order.

        Returns:
            Order: The created order object.
        """
        order = Order(
            order_id=self.next_order_id,
            instrument_id=instrument_id,
            price=price,
            volume=volume,
            side=side,
            order_type=order_type,
            user_id=user_id
        )
        if side.lower() == 'bid':
            color_msg = f"{Fore.BLUE}Inserted BUY {order_type.upper()} order: {order}{Style.RESET_ALL}"
        else:
            color_msg = f"{Fore.MAGENTA}Inserted SELL {order_type.upper()} order: {order}{Style.RESET_ALL}"
        logger.info(color_msg)
        self.next_order_id += 1

        if order_type.lower() == 'limit':
            # Add to order book
            self.orders[instrument_id][order.order_id] = order
            # Attempt to match
            self.match_orders(instrument_id)
        elif order_type.lower() == 'ioc':
            # Attempt to match immediately and execute partial fills
            filled_volume = self.match_orders(instrument_id, incoming_order=order)
            if filled_volume > 0:
                logger.info(
                    f"{Fore.GREEN}IOC order {order.order_id} partially filled with volume {filled_volume}.{Style.RESET_ALL}"
                )
            if order.volume > 0:
                # IOC order: Cancel any remaining volume after partial fill
                logger.info(
                    f"{Fore.RED}IOC order {order.order_id} canceled remaining volume {order.volume}.{Style.RESET_ALL}"
                )
        else:
            logger.warning(f"{Fore.YELLOW}Unknown order type: {order_type}. Order not processed.{Style.RESET_ALL}")

        return order

    def amend_order(self, instrument_id: str, order_id: int, price: Optional[float] = None, volume: Optional[int] = None):
        """
        Amend an existing order's price and/or volume.

        Args:
            instrument_id (str): The ID of the instrument.
            order_id (int): The ID of the order to amend.
            price (float, optional): The new price. Defaults to None.
            volume (int, optional): The new volume. Defaults to None.
        """
        order = self.orders[instrument_id].get(order_id)
        if order:
            original_order = Order(**vars(order))  # Create a copy for logging
            if price is not None:
                order.price = price
            if volume is not None:
                order.volume = volume
            logger.info(f"{Fore.CYAN}Amended order from {original_order} to {order}{Style.RESET_ALL}")
        else:
            logger.warning(f"{Fore.YELLOW}Order ID {order_id} not found for instrument {instrument_id}.{Style.RESET_ALL}")

    def delete_order(self, instrument_id: str, order_id: int):
        if order_id in self.orders[instrument_id]:
            del_order = self.orders[instrument_id].pop(order_id)
            logger.info(f"{Fore.RED}Deleted order: {del_order}{Style.RESET_ALL}")
        else:
            logger.warning(f"{Fore.YELLOW}Order ID {order_id} not found for instrument {instrument_id}.{Style.RESET_ALL}")

    def get_outstanding_orders(self, instrument_id: str) -> List[Order]:
        return list(self.orders[instrument_id].values())

    def get_trade_history(self, instrument_id: str) -> List[Trade]:
        return self.trade_history[instrument_id]

    def get_positions_and_cash(self) -> Dict[str, Any]:
        return {
            "positions": self.positions,
            "cash": self.cash
        }

    def match_orders(self, instrument_id: str, incoming_order: Optional[Order] = None) -> int:
        """
        Matches buy and sell orders for a given instrument.

        Args:
            instrument_id (str): The ID of the instrument to match orders for.
            incoming_order (Order, optional): The incoming IOC order to match. If None, match all possible.

        Returns:
            int: Total volume filled for the incoming IOC order.
        """
        total_filled = 0
        while True:
            bids = sorted(
                [o for o in self.orders[instrument_id].values() if o.side == 'bid'],
                key=lambda x: x.price,
                reverse=True
            )
            asks = sorted(
                [o for o in self.orders[instrument_id].values() if o.side == 'ask'],
                key=lambda x: x.price
            )

            if incoming_order:
                if incoming_order.side == 'bid':
                    if not asks:
                        break
                    lowest_ask = asks[0]
                    if incoming_order.price >= lowest_ask.price:
                        trade_price = (incoming_order.price + lowest_ask.price) / 2
                        trade_volume = min(incoming_order.volume, lowest_ask.volume)

                        # Update volumes
                        incoming_order.volume -= trade_volume
                        lowest_ask.volume -= trade_volume

                        # Remove ask order if filled
                        if lowest_ask.volume == 0:
                            del self.orders[instrument_id][lowest_ask.order_id]

                        # Record trade
                        trade = Trade(
                            trade_id=self.trade_id_counter,
                            instrument_id=instrument_id,
                            price=trade_price,
                            volume=trade_volume,
                            aggressor="buyer",
                            buyer_id=incoming_order.user_id,
                            seller_id=lowest_ask.user_id
                        )
                        self.trade_history[instrument_id].append(trade)
                        self.trade_id_counter += 1
                        logger.info(f"{Fore.GREEN}Trade executed: {trade}{Style.RESET_ALL}")

                        # Update positions and cash
                        self.positions[instrument_id] += trade_volume
                        self.cash -= trade_volume * trade_price

                        # Update price book
                        self.price_books[instrument_id].bid_prices.append(incoming_order.price)
                        self.price_books[instrument_id].ask_prices.append(lowest_ask.price)

                        total_filled += trade_volume

                        if incoming_order.volume == 0:
                            break
                    else:
                        break

                elif incoming_order.side == 'ask':
                    if not bids:
                        break
                    highest_bid = bids[0]
                    if incoming_order.price <= highest_bid.price:
                        trade_price = (highest_bid.price + incoming_order.price) / 2
                        trade_volume = min(highest_bid.volume, incoming_order.volume)

                        # Update volumes
                        highest_bid.volume -= trade_volume
                        incoming_order.volume -= trade_volume

                        # Remove bid order if filled
                        if highest_bid.volume == 0:
                            del self.orders[instrument_id][highest_bid.order_id]

                        # Record trade
                        trade = Trade(
                            trade_id=self.trade_id_counter,
                            instrument_id=instrument_id,
                            price=trade_price,
                            volume=trade_volume,
                            aggressor="seller",
                            buyer_id=highest_bid.user_id,
                            seller_id=incoming_order.user_id
                        )
                        self.trade_history[instrument_id].append(trade)
                        self.trade_id_counter += 1
                        logger.info(f"{Fore.GREEN}Trade executed: {trade}{Style.RESET_ALL}")

                        # Update positions and cash
                        self.positions[instrument_id] += trade_volume
                        self.cash -= trade_volume * trade_price

                        # Update price book
                        self.price_books[instrument_id].bid_prices.append(highest_bid.price)
                        self.price_books[instrument_id].ask_prices.append(incoming_order.price)

                        total_filled += trade_volume

                        if incoming_order.volume == 0:
                            break
                    else:
                        break
            else:
                if not bids or not asks:
                    break

                highest_bid = bids[0]
                lowest_ask = asks[0]

                if highest_bid.price >= lowest_ask.price:
                    trade_price = (highest_bid.price + lowest_ask.price) / 2
                    trade_volume = min(highest_bid.volume, lowest_ask.volume)

                    # Update volumes
                    highest_bid.volume -= trade_volume
                    lowest_ask.volume -= trade_volume

                    # Remove orders if filled
                    if highest_bid.volume == 0:
                        del self.orders[instrument_id][highest_bid.order_id]
                    if lowest_ask.volume == 0:
                        del self.orders[instrument_id][lowest_ask.order_id]

                    # Record trade
                    trade = Trade(
                        trade_id=self.trade_id_counter,
                        instrument_id=instrument_id,
                        price=trade_price,
                        volume=trade_volume,
                        aggressor="buyer",
                        buyer_id=highest_bid.user_id,
                        seller_id=lowest_ask.user_id
                    )
                    self.trade_history[instrument_id].append(trade)
                    self.trade_id_counter += 1
                    logger.info(f"{Fore.GREEN}Trade executed: {trade}{Style.RESET_ALL}")

                    # Update positions and cash
                    self.positions[instrument_id] += trade_volume
                    self.cash -= trade_volume * trade_price

                    # Update price book
                    self.price_books[instrument_id].bid_prices.append(highest_bid.price)
                    self.price_books[instrument_id].ask_prices.append(lowest_ask.price)
                else:
                    break

        return total_filled
