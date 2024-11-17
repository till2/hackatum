"""
Random Trading Strategy

This strategy places random buy and sell orders near the current market price.
If the price book is empty, it places orders near the last trade price.
If there is no trade history, it places orders at random prices between 0 and 1000.
"""

import logging
import math
import random
import os
from typing import Any

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RandomTradingStrategy:
    def __init__(
        self, 
        exchange: Any, 
        user_id: str, 
        instrument_id: str, 
        order_volume: int = 10, 
        price_range: float = 0.05
    ):
        """
        Initializes the Random Trading Strategy.

        Args:
            exchange (Exchange): The Exchange instance for market interaction.
            user_id (str): The user ID for placing orders.
            instrument_id (str): The ID of the instrument to trade.
            order_volume (int): The volume for each buy/sell order.
            price_range (float): The percentage range around the current price to place orders.
        """
        self.exchange = exchange
        self.user_id = user_id
        self.instrument_id = instrument_id
        self.order_volume = order_volume
        self.price_range = price_range

    def execute_strategy(self) -> None:
        """
        Executes the random trading strategy.
        """
        price_book = self.exchange.price_books.get(self.instrument_id)
        
        if price_book and price_book.bid_prices and price_book.ask_prices:
            # Calculate mid-price
            mid_price = (max(price_book.bid_prices) + min(price_book.ask_prices)) / 2.0
            logger.info(f"Current mid price for {self.instrument_id}: {mid_price}")
            self.place_random_orders(mid_price)
        else:
            # Handle empty price book
            trades = self.exchange.get_trade_history(self.instrument_id)
            if trades:
                # Use the last trade price
                last_trade = trades[-1]
                last_price = last_trade.price
                logger.info(f"Price book empty. Using last trade price: {last_price}")
                self.place_random_orders(last_price)
            else:
                # No trade history, place orders at random prices
                random_buy_price = random.uniform(0, 1000)
                random_sell_price = random.uniform(0, 1000)
                logger.info(f"Price book and trade history empty. Placing orders at random prices: Buy @ {random_buy_price}, Sell @ {random_sell_price}")
                self.place_order(random_buy_price, 'bid')
                self.place_order(random_sell_price, 'ask')

    def place_random_orders(self, reference_price: float) -> None:
        """
        Places buy and sell orders around the reference price within the specified range.

        Args:
            reference_price (float): The price around which to place orders.
        """
        # Calculate price offsets
        buy_offset = reference_price * self.price_range
        sell_offset = reference_price * self.price_range

        # Determine buy and sell prices
        buy_price = reference_price - buy_offset
        sell_price = reference_price + sell_offset

        # Round prices to nearest tick size (assuming 0.01)
        buy_price = math.floor(buy_price / 0.01) * 0.01
        sell_price = math.ceil(sell_price / 0.01) * 0.01

        logger.info(f"Placing buy order at {buy_price} and sell order at {sell_price} for {self.instrument_id}")
        
        self.place_order(buy_price, 'bid')
        self.place_order(sell_price, 'ask')

    def place_order(self, price: float, side: str) -> None:
        """
        Places a limit order with the specified price and side.

        Args:
            price (float): The price at which to place the order.
            side (str): 'bid' for buy orders or 'ask' for sell orders.
        """
        self.exchange.insert_order(
            instrument_id=self.instrument_id,
            price=price,
            volume=self.order_volume,
            side=side,
            order_type='limit',
            user_id=self.user_id
        )
        logger.info(f"Placed {side} order: {self.order_volume} @ {price}") 