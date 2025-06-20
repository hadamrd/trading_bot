import numpy as np
from typing import List, Dict

from tradingbot2.TradePosition import TradePosition
from .src.trading_bot.core.models import BacktestResult

class ResultsAnalyzer:
    @staticmethod
    def calculate_stats(trades: List[TradePosition]) -> Dict[str, float]:
        if not trades:
            return ResultsAnalyzer._empty_stats()

        profits = np.array([trade.profit for trade in trades])
        returns = np.array([trade.sell_return for trade in trades])

        winning_trades = profits > 0
        losing_trades = profits < 0

        total_profit = np.sum(profits)
        win_rate = np.mean(winning_trades)
        profit_factor = np.sum(profits[winning_trades]) / abs(np.sum(profits[losing_trades])) if np.any(losing_trades) else np.inf
        
        hold_times = np.array([trade.duration for trade in trades if trade.close_time is not None])

        stats = {
            "total_profit": total_profit,
            "avg_profit": np.mean(profits[winning_trades]) if np.any(winning_trades) else 0,
            "avg_loss": np.mean(profits[losing_trades]) if np.any(losing_trades) else 0,
            "std_profit": np.std(profits),
            "avg_hold_time": np.mean(hold_times) if len(hold_times) > 0 else 0,
            "std_hold_time": np.std(hold_times) if len(hold_times) > 0 else 0,
            "avg_return": np.mean(returns),
            "std_return": np.std(returns),
            "sharpe_ratio": ResultsAnalyzer.calculate_sharpe_ratio(returns),
            "sortino_ratio": ResultsAnalyzer.calculate_sortino_ratio(returns),
            "max_drawdown": ResultsAnalyzer.calculate_max_drawdown(returns),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "largest_win": np.max(profits) if np.any(winning_trades) else 0,
            "largest_loss": np.min(profits) if np.any(losing_trades) else 0,
            "total_trades": len(trades),
            "winning_trades": np.sum(winning_trades),
            "losing_trades": np.sum(losing_trades),
        }

        return stats

    @staticmethod
    def _empty_stats() -> Dict[str, float]:
        return {
            "total_profit": 0,
            "avg_profit": 0,
            "std_profit": 0,
            "avg_hold_time": 0,
            "std_hold_time": 0,
            "avg_return": 0,
            "avg_loss": 0,
            "std_return": 0,
            "sharpe_ratio": 0,
            "sortino_ratio": 0,
            "max_drawdown": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "largest_win": 0,
            "largest_loss": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
        }

    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        excess_returns = returns - risk_free_rate
        if len(excess_returns) < 2:
            return 0
        return np.mean(excess_returns) / np.std(excess_returns, ddof=1)

    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) < 2:
            return 0
        return np.mean(excess_returns) / np.std(downside_returns, ddof=1)

    @staticmethod
    def calculate_max_drawdown(returns: np.ndarray) -> float:
        cumulative_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        return np.max(drawdown)

    @staticmethod
    def generate_backtest_result(symbol: str, trades: List[TradePosition]) -> BacktestResult:
        stats = ResultsAnalyzer.calculate_stats(trades)
        return BacktestResult(
            symbol=symbol,
            total_trades=stats['total_trades'],
            winning_trades=stats['winning_trades'],
            losing_trades=stats['losing_trades'],
            total_profit=stats['total_profit'],
            win_rate=stats['win_rate'],
            average_profit=stats['avg_profit'],
            average_loss=abs(stats['avg_loss']),
            largest_win=stats['largest_win'],
            largest_loss=abs(stats['largest_loss']),
            average_holding_time=stats['avg_hold_time'],
            sharpe_ratio=stats['sharpe_ratio'],
            sortino_ratio=stats['sortino_ratio'],
            max_drawdown=stats['max_drawdown'],
            profit_factor=stats['profit_factor'],
            trades=trades
        )