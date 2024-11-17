"""
Mean Reversion Strategy

This strategy assumes that the price of an instrument will revert to its mean over time.
It identifies when the current price deviates significantly from the historical average and 
places orders anticipating a reversion.
"""

import logging
import math
import os
from typing import Any

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import calculate_moving_average

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MeanReversionStrategy:
    def __init__(self, exchange: Any, user_id: str, instrument_id: str, window_size: int = 20, threshold: float = 0.05):
        """
        Initializes the Mean Reversion Strategy.

        Args:
            exchange (Exchange): The Exchange instance for market interaction.
            user_id (str): The user ID for placing orders.
            instrument_id (str): The ID of the instrument to trade.
            window_size (int): Number of periods to calculate moving average.
            threshold (float): Percentage deviation from the moving average to trigger trades.
        """
        self.exchange = exchange
        self.user_id = user_id
        self.instrument_id = instrument_id
        self.window_size = window_size
        self.threshold = threshold
        self.price_history = []

    def update_price_history(self, current_price: float) -> None:
        """
        Updates the historical price data.

        Args:
            current_price (float): The latest price of the instrument.
        """
        self.price_history.append(current_price)
        if len(self.price_history) > self.window_size:
            self.price_history.pop(0)

    def execute_strategy(self) -> None:
        """
        Executes the mean reversion trading strategy.
        """
        price_book = self.exchange.price_books.get(self.instrument_id)
        if not price_book or not price_book.bid_prices or not price_book.ask_prices:
            logger.warning(f"Empty price book for {self.instrument_id}. Skipping strategy execution.")
            return

        mid_price = (max(price_book.bid_prices) + min(price_book.ask_prices)) / 2.0
        self.update_price_history(mid_price)

        if len(self.price_history) < self.window_size:
            logger.info("Insufficient price history. Waiting to accumulate data.")
            return

        moving_avg = calculate_moving_average(self.price_history)

        deviation = (mid_price - moving_avg) / moving_avg

        logger.info(f"Mid Price: {mid_price}, Moving Avg: {moving_avg}, Deviation: {deviation:.2%}")

        # Trigger buy if price is below the moving average beyond the threshold
        if deviation < -self.threshold:
            logger.info("Price below moving average by threshold. Placing buy orders.")
            self.place_buy_order(mid_price)

        # Trigger sell if price is above the moving average beyond the threshold
        elif deviation > self.threshold:
            logger.info("Price above moving average by threshold. Placing sell orders.")
            self.place_sell_order(mid_price)

    def place_buy_order(self, price: float, volume: int = 10) -> None:
        """
        Places a buy limit order below the current market bid.

        Args:
            price (float): The price at which to place the buy order.
            volume (int): The volume to buy.
        """
        buy_price = price * (1 - self.threshold)
        buy_price = math.floor(buy_price / 0.01) * 0.01  # Round down to nearest tick size (assuming 0.01)
        self.exchange.insert_order(
            instrument_id=self.instrument_id,
            price=buy_price,
            volume=volume,
            side='bid',
            order_type='limit',
            user_id=self.user_id
        )
        logger.info(f"Placed buy order: {volume} @ {buy_price}")

    def place_sell_order(self, price: float, volume: int = 10) -> None:
        """
        Places a sell limit order above the current market ask.

        Args:
            price (float): The price at which to place the sell order.
            volume (int): The volume to sell.
        """
        sell_price = price * (1 + self.threshold)
        sell_price = math.ceil(sell_price / 0.01) * 0.01  # Round up to nearest tick size (assuming 0.01)
        self.exchange.insert_order(
            instrument_id=self.instrument_id,
            price=sell_price,
            volume=volume,
            side='ask',
            order_type='limit',
            user_id=self.user_id
        )
        logger.info(f"Placed sell order: {volume} @ {sell_price}") 