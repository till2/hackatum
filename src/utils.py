"""
Utility Functions

Provides helper functions for trading strategies, including moving averages, rate of change,
trend slope calculations, and spread statistics.
"""

import math
from typing import List, Tuple


def calculate_moving_average(price_history: List[float]) -> float:
    """
    Calculates the moving average of the given price history.

    Args:
        price_history (List[float]): List of historical prices.

    Returns:
        float: The moving average.
    """
    return sum(price_history) / len(price_history)


def calculate_rate_of_change(price_history: List[float]) -> float:
    """
    Calculates the rate of change (ROC) of the price history.

    Args:
        price_history (List[float]): List of historical prices.

    Returns:
        float: The rate of change.
    """
    start_price = price_history[0]
    end_price = price_history[-1]
    return (end_price - start_price) / start_price


def calculate_slope(price_history: List[float]) -> float:
    """
    Calculates the slope of the price trend using linear regression.

    Args:
        price_history (List[float]): List of historical prices.

    Returns:
        float: The slope of the trend.
    """
    n = len(price_history)
    x = list(range(n))
    y = price_history
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xx = sum([i**2 for i in x])
    sum_xy = sum([xi * yi for xi, yi in zip(x, y)])
    denominator = n * sum_xx - sum_x ** 2
    if denominator == 0:
        return 0.0
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return slope


def calculate_spread(spread_history: List[float]) -> Tuple[float, float]:
    """
    Calculates the mean and standard deviation of the spread history.

    Args:
        spread_history (List[float]): List of historical spreads.

    Returns:
        Tuple[float, float]: The mean and standard deviation of the spread.
    """
    mean = sum(spread_history) / len(spread_history)
    variance = sum([(s - mean) ** 2 for s in spread_history]) / len(spread_history)
    std_dev = math.sqrt(variance)
    return mean, std_dev


def calculate_z_score(spread: float, mean: float, std_dev: float) -> float:
    """
    Calculates the z-score of the current spread.

    Args:
        spread (float): The current spread.
        mean (float): The mean of the spread history.
        std_dev (float): The standard deviation of the spread history.

    Returns:
        float: The z-score.
    """
    if std_dev == 0:
        return 0.0
    return (spread - mean) / std_dev 