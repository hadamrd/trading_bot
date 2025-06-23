# src/trading_bot/optimization/__init__.py
"""
Optimization package for trading strategies.
"""

from .genetic import GeneticOptimizer, StrategyEvaluator, optimize_strategy

__all__ = [
    "GeneticOptimizer",
    "StrategyEvaluator", 
    "optimize_strategy"
]