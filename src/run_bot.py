import time
import logging
import threading
from common_types import Order, PriceBook, Trade
from synchronous_client import Exchange
from strategies import (
    ArbitrageStrategy,
    MeanReversionStrategy,
    MomentumTradingStrategy,
    StatisticalArbitrageStrategy,
    TrendFollowingStrategy,
    RandomTradingStrategy
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

time_between_executions = 1

def run_strategy(strategy, interval):
    """Function to run a trading strategy in a loop."""
    try:
        while True:
            strategy.execute_strategy()
            time.sleep(interval)
    except Exception as e:
        logger.error(f"Error in {strategy.__class__.__name__}: {e}")

def main():
    # Initialize Exchange
    exchange = Exchange()
    exchange.connect()

    # Initialize the price book
    exchange.price_books["AAPL"] = PriceBook(
        bid_prices=[145.0, 147.0, 149.0, 151.0, 153.0],
        ask_prices=[146.0, 148.0, 150.0, 152.0, 154.0]
    )

    # Initialize Mean Reversion Strategy
    mean_reversion_strategy = MeanReversionStrategy(
        exchange=exchange,
        user_id="mean_reversion_trader",
        instrument_id="AAPL",
        window_size=20,
        threshold=0.05
    )
    logger.info("Mean Reversion Strategy Initialized.")

    # Initialize Random Trading Strategy
    random_trading_strategy = RandomTradingStrategy(
        exchange=exchange,
        user_id="random_trader",
        instrument_id="AAPL"
    )
    logger.info("Random Trading Strategy Initialized.")

    # Create threads for each strategy
    mean_reversion_thread = threading.Thread(
        target=run_strategy,
        args=(mean_reversion_strategy, time_between_executions),
        name="MeanReversionThread",
        daemon=True
    )

    random_trading_thread = threading.Thread(
        target=run_strategy,
        args=(random_trading_strategy, time_between_executions),
        name="RandomTradingThread",
        daemon=True
    )

    logger.info("Starting Strategy Execution Threads.")

    # Start the strategy threads
    mean_reversion_thread.start()
    random_trading_thread.start()

    # Keep the main thread alive to allow strategies to run
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Strategy execution terminated by user.")
    finally:
        exchange.disconnect()
        logger.info("Exchange connection closed.")

if __name__ == "__main__":
    main() 