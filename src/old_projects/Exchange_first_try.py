import logging
from typing import List, Dict, Optional

from common_types import Order, PriceBook, Trade, SocialMediaFeed, Instrument
from synchronous_client import ExchangeClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Exchange:
    def __init__(self, host: str = 'localhost', port: int = 38200):
        """
        Initializes the Exchange simulator instance.

        Args:
            host (str): The hostname (unused in simulation).
            port (int): The port number (unused in simulation).
        """
        self.exchange = ExchangeClient()

    def connect(self) -> None:
        """
        Establishes a connection to the Exchange simulator.
        """
        try:
            self.exchange.connect()
            logger.info("Connected to Exchange Simulator.")
        except Exception as e:
            logger.error(f"Failed to connect to Exchange Simulator: {e}")
            raise

    def disconnect(self) -> None:
        """
        Disconnects from the Exchange simulator.
        """
        try:
            self.exchange.disconnect()
            logger.info("Disconnected from Exchange Simulator.")
        except Exception as e:
            logger.error(f"Failed to disconnect from Exchange Simulator: {e}")
            raise

    def get_instruments(self) -> Dict[str, 'Instrument']:
        """
        Retrieves all instruments available on the exchange.

        Returns:
            Dict[str, Instrument]: A dictionary mapping instrument IDs to Instrument objects.
        """
        instruments = self.exchange.get_instruments()
        # For simulation, we'll represent instruments as empty dicts or simple objects
        instrument_dict = {inst: {"instrument_id": inst} for inst in instruments}
        return instrument_dict

    def get_last_price_book(self, instrument_id: str) -> Optional[PriceBookUpdate]:
        """
        Fetches the latest price book for a given instrument.

        Args:
            instrument_id (str): The ID of the instrument.

        Returns:
            Optional[PriceBookUpdate]: The latest PriceBook or None if unavailable.
        """
        try:
            price_book = self.exchange.get_last_price_book(instrument_id)
            return price_book
        except Exception as e:
            logger.error(f"Error fetching price book for {instrument_id}: {e}")
            return None

    def insert_order(
        self,
        instrument_id: str,
        price: float,
        volume: int,
        side: str,
        order_type: str = 'limit'
    ) -> Optional[Order]:
        """
        Inserts a new order into the exchange.

        Args:
            instrument_id (str): The ID of the instrument.
            price (float): The price for the order.
            volume (int): The volume for the order.
            side (str): 'bid' or 'ask'.
            order_type (str): Type of the order, default is 'limit'.

        Returns:
            Optional[Order]: The inserted Order object or None if unsuccessful.
        """
        try:
            order = self.exchange.insert_order(
                instrument_id=instrument_id,
                price=price,
                volume=volume,
                side=side,
                order_type=order_type
            )
            logger.info(f"Inserted {side} order: {order}")
            return order
        except Exception as e:
            logger.error(f"Failed to insert {side} order for {instrument_id}: {e}")
            return None

    def delete_order(self, instrument_id: str, order_id: int) -> bool:
        """
        Deletes an existing order from the exchange.

        Args:
            instrument_id (str): The ID of the instrument.
            order_id (int): The ID of the order to delete.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            result = self.exchange.delete_order(instrument_id, order_id)
            if result:
                logger.info(f"Deleted order {order_id} for {instrument_id}.")
            else:
                logger.warning(f"Order {order_id} for {instrument_id} could not be deleted.")
            return result
        except Exception as e:
            logger.error(f"Failed to delete order {order_id} for {instrument_id}: {e}")
            return False

    def amend_order(
        self,
        instrument_id: str,
        order_id: int,
        price: Optional[float] = None,
        volume: Optional[int] = None
    ) -> bool:
        """
        Amends an existing order's price and/or volume.

        Args:
            instrument_id (str): The ID of the instrument.
            order_id (int): The ID of the order to amend.
            price (Optional[float]): The new price for the order.
            volume (Optional[int]): The new volume for the order.

        Returns:
            bool: True if amendment was successful, False otherwise.
        """
        try:
            result = self.exchange.amend_order(
                instrument_id=instrument_id,
                order_id=order_id,
                price=price,
                volume=volume
            )
            if result:
                logger.info(f"Amended order {order_id} for {instrument_id}.")
            else:
                logger.warning(f"Order {order_id} for {instrument_id} could not be amended.")
            return result
        except Exception as e:
            logger.error(f"Failed to amend order {order_id} for {instrument_id}: {e}")
            return False

    def get_positions(self) -> Dict[str, int]:
        """
        Retrieves the current positions for all instruments.

        Returns:
            Dict[str, int]: A dictionary mapping instrument IDs to their current positions.
        """
        try:
            positions = self.exchange.get_positions()
            return positions
        except Exception as e:
            logger.error(f"Error retrieving positions: {e}")
            return {}

    def get_positions_and_cash(self) -> Dict[str, float]:
        """
        Retrieves current positions along with cash invested.

        Returns:
            Dict[str, float]: A dictionary mapping instrument IDs and 'cash' to their respective values.
        """
        try:
            positions_cash = self.exchange.get_positions_and_cash()
            return positions_cash
        except Exception as e:
            logger.error(f"Error retrieving positions and cash: {e}")
            return {}

    def get_trade_history(self, instrument_id: str) -> List[Trade]:
        """
        Retrieves the trade history for a specific instrument.

        Args:
            instrument_id (str): The ID of the instrument.

        Returns:
            List[Trade]: A list of Trade objects.
        """
        try:
            trades = self.exchange.get_trade_history(instrument_id)
            return trades
        except Exception as e:
            logger.error(f"Error fetching trade history for {instrument_id}: {e}")
            return []

    def get_outstanding_orders(self, instrument_id: str) -> Dict[int, Order]:
        """
        Retrieves all outstanding orders for a specific instrument.

        Args:
            instrument_id (str): The ID of the instrument.

        Returns:
            Dict[int, Order]: A dictionary mapping order IDs to Order objects.
        """
        try:
            orders = self.exchange.get_outstanding_orders(instrument_id)
            return orders
        except Exception as e:
            logger.error(f"Error fetching outstanding orders for {instrument_id}: {e}")
            return {}

    def poll_new_social_media_feeds(self, continuous: bool = False) -> List[SocialMediaFeed]:
        """
        Polls new social media feeds.

        Args:
            continuous (bool): If True, keeps polling indefinitely.

        Returns:
            List[SocialMediaFeed]: A list of new SocialMediaFeed objects.
        """
        try:
            feeds = self.exchange.poll_new_social_media_feeds(continuous=continuous)
            return feeds
        except Exception as e:
            logger.error(f"Error polling social media feeds: {e}")
            return [] 