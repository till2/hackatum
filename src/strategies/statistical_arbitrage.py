"""
Statistical Arbitrage Strategy

This strategy uses statistical methods to identify and exploit price inefficiencies between related instruments.
It relies on mean-reverting spreads and pairs trading to capture profits from price convergence.
"""

import logging
import math
import os
from typing import Any

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import calculate_spread, calculate_z_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalArbitrageStrategy:
    def __init__(
        self, 
        exchange: Any, 
        user_id: str, 
        instrument_a: str, 
        instrument_b: str, 
        window_size: int = 30, 
        entry_threshold: float = 2.0, 
        exit_threshold: float = 0.5
    ):
        """
        Initializes the Statistical Arbitrage Strategy.

        Args:
            exchange (Exchange): The Exchange instance for market interaction.
            user_id (str): The user ID for placing orders.
            instrument_a (str): The ID of the first instrument.
            instrument_b (str): The ID of the second instrument.
            window_size (int): Number of periods to calculate spread statistics.
            entry_threshold (float): Z-score threshold to enter a trade.
            exit_threshold (float): Z-score threshold to exit a trade.
        """
        self.exchange = exchange
        self.user_id = user_id
        self.instrument_a = instrument_a
        self.instrument_b = instrument_b
        self.window_size = window_size
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.spread_history = []

    def update_spread_history(self, spread: float) -> None:
        """
        Updates the historical spread data.

        Args:
            spread (float): The current spread between instrument A and B.
        """
        self.spread_history.append(spread)
        if len(self.spread_history) > self.window_size:
            self.spread_history.pop(0)

    def execute_strategy(self) -> None:
        """
        Executes the statistical arbitrage trading strategy.
        """
        price_book_a = self.exchange.price_books.get(self.instrument_a)
        price_book_b = self.exchange.price_books.get(self.instrument_b)

        if not price_book_a or not price_book_a.bid_prices or not price_book_a.ask_prices:
            logger.warning(f"Incomplete price book for {self.instrument_a}. Skipping strategy execution.")
            return

        if not price_book_b or not price_book_b.bid_prices or not price_book_b.ask_prices:
            logger.warning(f"Incomplete price book for {self.instrument_b}. Skipping strategy execution.")
            return

        mid_a = (max(price_book_a.bid_prices) + min(price_book_a.ask_prices)) / 2.0
        mid_b = (max(price_book_b.bid_prices) + min(price_book_b.ask_prices)) / 2.0

        spread = mid_a - mid_b
        self.update_spread_history(spread)

        if len(self.spread_history) < self.window_size:
            logger.info("Insufficient spread history. Waiting to accumulate data.")
            return

        mean, std = calculate_spread(self.spread_history)
        z_score = calculate_z_score(spread, mean, std)

        logger.info(f"Spread: {spread:.2f}, Mean: {mean:.2f}, Std: {std:.2f}, Z-score: {z_score:.2f}")

        # Enter trades based on z-score
        if z_score > self.entry_threshold:
            logger.info("Spread above entry threshold. Entering short spread position.")
            self.enter_short_spread()
        elif z_score < -self.entry_threshold:
            logger.info("Spread below entry threshold. Entering long spread position.")
            self.enter_long_spread()

        # Exit trades based on z-score
        elif abs(z_score) < self.exit_threshold:
            logger.info("Spread within exit threshold. Exiting any open spread positions.")
            self.exit_spread()

    def enter_long_spread(self, volume: int = 10) -> None:
        """
        Enters a long spread position by buying Instrument A and selling Instrument B.

        Args:
            volume (int): The volume to trade.
        """
        # Buy Instrument A
        buy_price = max(self.exchange.price_books[self.instrument_a].bid_prices)
        self.exchange.insert_order(
            instrument_id=self.instrument_a,
            price=buy_price,
            volume=volume,
            side='bid',
            order_type='limit',
            user_id=self.user_id
        )
        logger.info(f"Placed buy order: {volume} @ {buy_price} on {self.instrument_a}")

        # Sell Instrument B
        sell_price = min(self.exchange.price_books[self.instrument_b].ask_prices)
        self.exchange.insert_order(
            instrument_id=self.instrument_b,
            price=sell_price,
            volume=volume,
            side='ask',
            order_type='limit',
            user_id=self.user_id
        )
        logger.info(f"Placed sell order: {volume} @ {sell_price} on {self.instrument_b}")

    def enter_short_spread(self, volume: int = 10) -> None:
        """
        Enters a short spread position by selling Instrument A and buying Instrument B.

        Args:
            volume (int): The volume to trade.
        """
        # Sell Instrument A
        sell_price = min(self.exchange.price_books[self.instrument_a].ask_prices)
        self.exchange.insert_order(
            instrument_id=self.instrument_a,
            price=sell_price,
            volume=volume,
            side='ask',
            order_type='limit',
            user_id=self.user_id
        )
        logger.info(f"Placed sell order: {volume} @ {sell_price} on {self.instrument_a}")

        # Buy Instrument B
        buy_price = max(self.exchange.price_books[self.instrument_b].bid_prices)
        self.exchange.insert_order(
            instrument_id=self.instrument_b,
            price=buy_price,
            volume=volume,
            side='bid',
            order_type='limit',
            user_id=self.user_id
        )
        logger.info(f"Placed buy order: {volume} @ {buy_price} on {self.instrument_b}")

    def exit_spread(self, volume: int = 10) -> None:
        """
        Exits any open spread positions by canceling outstanding orders.

        Args:
            volume (int): The volume to trade.
        """
        outstanding_a = self.exchange.get_outstanding_orders(self.instrument_a)
        outstanding_b = self.exchange.get_outstanding_orders(self.instrument_b)

        for order in outstanding_a:
            self.exchange.delete_order(self.instrument_a, order.order_id)
            logger.info(f"Canceled order {order.order_id} on {self.instrument_a}")

        for order in outstanding_b:
            self.exchange.delete_order(self.instrument_b, order.order_id)
            logger.info(f"Canceled order {order.order_id} on {self.instrument_b}") 