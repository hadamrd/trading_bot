"""
Trading strategies package.
"""

from .base import BaseStrategy
from .ema_crossover import EMACrossoverStrategy

__all__ = [
    "BaseStrategy",
    "EMACrossoverStrategy",
]