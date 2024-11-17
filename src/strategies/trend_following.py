"""
Trend Following Strategy

This strategy identifies sustained movements in price direction (uptrend or downtrend)
and places orders to capitalize on the continuation of these trends.
"""

import logging
import math
import os
from typing import Any

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import calculate_slope

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrendFollowingStrategy:
    def __init__(self, exchange: Any, user_id: str, instrument_id: str, period: int = 15, slope_threshold: float = 0.01):
        """
        Initializes the Trend Following Strategy.

        Args:
            exchange (Exchange): The Exchange instance for market interaction.
            user_id (str): The user ID for placing orders.
            instrument_id (str): The ID of the instrument to trade.
            period (int): Number of periods to calculate the trend slope.
            slope_threshold (float): Slope threshold to trigger trades.
        """
        self.exchange = exchange
        self.user_id = user_id
        self.instrument_id = instrument_id
        self.period = period
        self.slope_threshold = slope_threshold
        self.price_history = []

    def update_price_history(self, current_price: float) -> None:
        """
        Updates the historical price data.

        Args:
            current_price (float): The latest price of the instrument.
        """
        self.price_history.append(current_price)
        if len(self.price_history) > self.period:
            self.price_history.pop(0)

    def execute_strategy(self) -> None:
        """
        Executes the trend following trading strategy.
        """
        price_book = self.exchange.price_books.get(self.instrument_id)
        if not price_book or not price_book.bid_prices or not price_book.ask_prices:
            logger.warning(f"Incomplete price book for {self.instrument_id}. Skipping strategy execution.")
            return

        mid_price = (max(price_book.bid_prices) + min(price_book.ask_prices)) / 2.0
        self.update_price_history(mid_price)

        if len(self.price_history) < self.period:
            logger.info("Insufficient price history. Waiting to accumulate data.")
            return

        slope = calculate_slope(self.price_history)

        logger.info(f"Trend Slope: {slope:.4f}")

        # Trigger buy if the slope is positive beyond the threshold
        if slope > self.slope_threshold:
            logger.info("Positive trend detected. Placing buy orders.")
            self.place_buy_order(mid_price)

        # Trigger sell if the slope is negative beyond the threshold
        elif slope < -self.slope_threshold:
            logger.info("Negative trend detected. Placing sell orders.")
            self.place_sell_order(mid_price)

    def place_buy_order(self, price: float, volume: int = 20) -> None:
        """
        Places a buy limit order above the current market bid.

        Args:
            price (float): The price at which to place the buy order.
            volume (int): The volume to buy.
        """
        buy_price = price * 1.01  # 1% above current price
        buy_price = math.floor(buy_price / 0.01) * 0.01  # Round down to nearest tick size
        self.exchange.insert_order(
            instrument_id=self.instrument_id,
            price=buy_price,
            volume=volume,
            side='bid',
            order_type='limit',
            user_id=self.user_id
        )
        logger.info(f"Placed buy order: {volume} @ {buy_price}")

    def place_sell_order(self, price: float, volume: int = 20) -> None:
        """
        Places a sell limit order below the current market ask.

        Args:
            price (float): The price at which to place the sell order.
            volume (int): The volume to sell.
        """
        sell_price = price * 0.99  # 1% below current price
        sell_price = math.ceil(sell_price / 0.01) * 0.01  # Round up to nearest tick size
        self.exchange.insert_order(
            instrument_id=self.instrument_id,
            price=sell_price,
            volume=volume,
            side='ask',
            order_type='limit',
            user_id=self.user_id
        )
        logger.info(f"Placed sell order: {volume} @ {sell_price}") 