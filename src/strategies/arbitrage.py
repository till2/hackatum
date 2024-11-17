"""
Arbitrage Strategy

This strategy exploits price discrepancies between different markets or instruments.
It simultaneously buys low in one market and sells high in another to lock in risk-free profits.
"""

import logging
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArbitrageStrategy:
    def __init__(self, exchange: Any, user_id: str, instrument_a: str, instrument_b: str, threshold: float = 0.02):
        """
        Initializes the Arbitrage Strategy.

        Args:
            exchange (Exchange): The Exchange instance for market interaction.
            user_id (str): The user ID for placing orders.
            instrument_a (str): The ID of the first instrument.
            instrument_b (str): The ID of the second instrument.
            threshold (float): The minimum price difference to trigger arbitrage.
        """
        self.exchange = exchange
        self.user_id = user_id
        self.instrument_a = instrument_a
        self.instrument_b = instrument_b
        self.threshold = threshold

    def execute_strategy(self) -> None:
        """
        Executes the arbitrage trading strategy.
        """
        price_book_a = self.exchange.price_books.get(self.instrument_a)
        price_book_b = self.exchange.price_books.get(self.instrument_b)

        if not price_book_a or not price_book_a.bid_prices or not price_book_a.ask_prices:
            logger.warning(f"Incomplete price book for {self.instrument_a}. Skipping arbitrage execution.")
            return

        if not price_book_b or not price_book_b.bid_prices or not price_book_b.ask_prices:
            logger.warning(f"Incomplete price book for {self.instrument_b}. Skipping arbitrage execution.")
            return

        ask_a = min(price_book_a.ask_prices)
        bid_b = max(price_book_b.bid_prices)

        logger.info(f"{self.instrument_a} Ask: {ask_a}, {self.instrument_b} Bid: {bid_b}")

        price_diff = bid_b - ask_a

        if price_diff > self.threshold:
            logger.info(f"Arbitrage opportunity detected! Price difference: {price_diff:.2f}")
            self.place_arbitrage_orders(ask_a, bid_b)
        else:
            logger.info("No arbitrage opportunity detected.")

    def place_arbitrage_orders(self, buy_price: float, sell_price: float, volume: int = 10) -> None:
        """
        Places buy and sell orders to exploit the arbitrage opportunity.

        Args:
            buy_price (float): The price at which to buy the first instrument.
            sell_price (float): The price at which to sell the second instrument.
            volume (int): The volume to trade.
        """
        # Place buy order on Instrument A
        self.exchange.insert_order(
            instrument_id=self.instrument_a,
            price=buy_price,
            volume=volume,
            side='bid',
            order_type='limit',
            user_id=self.user_id
        )
        logger.info(f"Placed buy order: {volume} @ {buy_price} on {self.instrument_a}")

        # Place sell order on Instrument B
        self.exchange.insert_order(
            instrument_id=self.instrument_b,
            price=sell_price,
            volume=volume,
            side='ask',
            order_type='limit',
            user_id=self.user_id
        )
        logger.info(f"Placed sell order: {volume} @ {sell_price} on {self.instrument_b}") 