"""
Advanced Analysis Module for Trading Bot Results
Provides comprehensive statistics, visualizations, and reporting
"""

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

from ..core.models import BacktestResult, Position
from ..core.enums import TradeStatus
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TradeAnalyzer:
    """Analyze individual trades and trade sequences"""
    
    @staticmethod
    def analyze_trades(trades: List[Position]) -> Dict[str, Any]:
        """
        Comprehensive analysis of trade list
        
        Args:
            trades: List of completed trades
            
        Returns:
            Dictionary with trade statistics
        """
        if not trades:
            return TradeAnalyzer._empty_trade_stats()
        
        # Convert to arrays for efficient computation
        profits = np.array([float(trade.profit) for trade in trades])
        returns = np.array([trade.return_percentage / 100.0 for trade in trades])
        hold_times = np.array([trade.duration_hours for trade in trades])
        
        # Basic statistics
        winning_trades = profits > 0
        losing_trades = profits < 0
        
        total_profit = np.sum(profits)
        win_rate = np.mean(winning_trades)
        total_trades = len(trades)
        
        # Profit statistics
        avg_win = np.mean(profits[winning_trades]) if np.any(winning_trades) else 0
        avg_loss = np.mean(profits[losing_trades]) if np.any(losing_trades) else 0
        largest_win = np.max(profits) if np.any(winning_trades) else 0
        largest_loss = np.min(profits) if np.any(losing_trades) else 0
        
        # Return statistics
        avg_return = np.mean(returns)
        std_return = np.std(returns, ddof=1) if len(returns) > 1 else 0
        
        # Risk metrics
        sharpe_ratio = TradeAnalyzer.calculate_sharpe_ratio(returns)
        sortino_ratio = TradeAnalyzer.calculate_sortino_ratio(returns)
        calmar_ratio = TradeAnalyzer.calculate_calmar_ratio(returns)
        max_drawdown = TradeAnalyzer.calculate_max_drawdown(returns)
        
        # Profit factor
        gross_profit = np.sum(profits[winning_trades]) if np.any(winning_trades) else 0
        gross_loss = abs(np.sum(profits[losing_trades])) if np.any(losing_trades) else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Holding time statistics
        avg_hold_time = np.mean(hold_times)
        std_hold_time = np.std(hold_times, ddof=1) if len(hold_times) > 1 else 0
        
        # Consecutive analysis
        consecutive_stats = TradeAnalyzer._analyze_consecutive_trades(profits)
        
        # Monthly/weekly analysis
        time_analysis = TradeAnalyzer._analyze_time_patterns(trades)
        
        return {
            # Basic metrics
            'total_trades': total_trades,
            'winning_trades': int(np.sum(winning_trades)),
            'losing_trades': int(np.sum(losing_trades)),
            'win_rate': win_rate,
            
            # Profit metrics
            'total_profit': total_profit,
            'avg_profit_per_trade': total_profit / total_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            
            # Return metrics
            'avg_return': avg_return,
            'std_return': std_return,
            'total_return': np.sum(returns),
            
            # Risk metrics
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            
            # Time metrics
            'avg_hold_time_hours': avg_hold_time,
            'std_hold_time_hours': std_hold_time,
            'min_hold_time_hours': np.min(hold_times),
            'max_hold_time_hours': np.max(hold_times),
            
            # Advanced metrics
            'expectancy': avg_return,  # Expected return per trade
            'system_quality_number': TradeAnalyzer.calculate_sqn(returns),
            'recovery_factor': abs(total_profit / largest_loss) if largest_loss < 0 else np.inf,
            
            # Consecutive analysis
            **consecutive_stats,
            
            # Time pattern analysis
            **time_analysis
        }
    
    @staticmethod
    def _empty_trade_stats() -> Dict[str, Any]:
        """Return empty statistics for no trades"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_profit': 0.0,
            'avg_profit_per_trade': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'profit_factor': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'avg_return': 0.0,
            'std_return': 0.0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_hold_time_hours': 0.0,
            'std_hold_time_hours': 0.0,
            'min_hold_time_hours': 0.0,
            'max_hold_time_hours': 0.0,
            'expectancy': 0.0,
            'system_quality_number': 0.0,
            'recovery_factor': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'avg_consecutive_wins': 0.0,
            'avg_consecutive_losses': 0.0
        }
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - risk_free_rate
        if np.std(excess_returns, ddof=1) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(excess_returns, ddof=1)
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) < 2:
            return 0.0
        downside_std = np.std(downside_returns, ddof=1)
        if downside_std == 0:
            return 0.0
        return np.mean(excess_returns) / downside_std
    
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        if len(returns) == 0:
            return 0.0
        annual_return = np.mean(returns) * 252  # Assuming daily returns
        max_dd = TradeAnalyzer.calculate_max_drawdown(returns)
        if max_dd == 0:
            return np.inf if annual_return > 0 else 0.0
        return annual_return / max_dd
    
    @staticmethod
    def calculate_max_drawdown(returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0.0
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(np.min(drawdown))
    
    @staticmethod
    def calculate_sqn(returns: np.ndarray) -> float:
        """Calculate System Quality Number (Van Tharp)"""
        if len(returns) < 2:
            return 0.0
        if np.std(returns, ddof=1) == 0:
            return 0.0
        return np.sqrt(len(returns)) * np.mean(returns) / np.std(returns, ddof=1)
    
    @staticmethod
    def _analyze_consecutive_trades(profits: np.ndarray) -> Dict[str, Any]:
        """Analyze consecutive wins/losses"""
        if len(profits) == 0:
            return {
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'avg_consecutive_wins': 0.0,
                'avg_consecutive_losses': 0.0
            }
        
        # Find consecutive sequences
        wins = profits > 0
        losses = profits <= 0
        
        # Count consecutive wins
        win_sequences = []
        current_win_streak = 0
        for is_win in wins:
            if is_win:
                current_win_streak += 1
            else:
                if current_win_streak > 0:
                    win_sequences.append(current_win_streak)
                current_win_streak = 0
        if current_win_streak > 0:
            win_sequences.append(current_win_streak)
        
        # Count consecutive losses
        loss_sequences = []
        current_loss_streak = 0
        for is_loss in losses:
            if is_loss:
                current_loss_streak += 1
            else:
                if current_loss_streak > 0:
                    loss_sequences.append(current_loss_streak)
                current_loss_streak = 0
        if current_loss_streak > 0:
            loss_sequences.append(current_loss_streak)
        
        return {
            'max_consecutive_wins': max(win_sequences) if win_sequences else 0,
            'max_consecutive_losses': max(loss_sequences) if loss_sequences else 0,
            'avg_consecutive_wins': np.mean(win_sequences) if win_sequences else 0.0,
            'avg_consecutive_losses': np.mean(loss_sequences) if loss_sequences else 0.0
        }
    
    @staticmethod
    def _analyze_time_patterns(trades: List[Position]) -> Dict[str, Any]:
        """Analyze time-based patterns in trading"""
        if not trades:
            return {}
        
        # Extract time information
        entry_times = [trade.open_time for trade in trades]
        profits = [float(trade.profit) for trade in trades]
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame({
            'entry_time': entry_times,
            'profit': profits,
            'hour': [t.hour for t in entry_times],
            'day_of_week': [t.weekday() for t in entry_times],  # 0=Monday
            'month': [t.month for t in entry_times]
        })
        
        # Hourly analysis
        hourly_stats = df.groupby('hour')['profit'].agg(['count', 'mean', 'sum']).to_dict()
        
        # Daily analysis
        daily_stats = df.groupby('day_of_week')['profit'].agg(['count', 'mean', 'sum']).to_dict()
        
        # Monthly analysis
        monthly_stats = df.groupby('month')['profit'].agg(['count', 'mean', 'sum']).to_dict()
        
        return {
            'hourly_performance': hourly_stats,
            'daily_performance': daily_stats,
            'monthly_performance': monthly_stats,
            'best_trading_hour': df.groupby('hour')['profit'].mean().idxmax() if len(df) > 0 else None,
            'best_trading_day': df.groupby('day_of_week')['profit'].mean().idxmax() if len(df) > 0 else None,
            'best_trading_month': df.groupby('month')['profit'].mean().idxmax() if len(df) > 0 else None
        }


class PortfolioAnalyzer:
    """Analyze portfolio-level performance across multiple strategies/symbols"""
    
    @staticmethod
    def analyze_portfolio(results: Dict[str, BacktestResult]) -> Dict[str, Any]:
        """
        Analyze portfolio performance across multiple symbols/strategies
        
        Args:
            results: Dictionary of symbol -> BacktestResult
            
        Returns:
            Portfolio-level statistics
        """
        if not results:
            return PortfolioAnalyzer._empty_portfolio_stats()
        
        # Aggregate statistics
        total_trades = sum(result.total_trades for result in results.values())
        total_profit = sum(float(result.total_profit) for result in results.values())
        total_return_pct = sum(result.total_return_pct for result in results.values())
        
        # Weighted averages
        weights = [result.total_trades for result in results.values()]
        total_weight = sum(weights)
        
        if total_weight > 0:
            weighted_win_rate = sum(result.win_rate * w for result, w in zip(results.values(), weights)) / total_weight
            weighted_sharpe = sum(result.sharpe_ratio * w for result, w in zip(results.values(), weights)) / total_weight
            weighted_max_dd = sum(result.max_drawdown * w for result, w in zip(results.values(), weights)) / total_weight
        else:
            weighted_win_rate = 0.0
            weighted_sharpe = 0.0
            weighted_max_dd = 0.0
        
        # Best/worst performers
        best_performer = max(results.items(), key=lambda x: x[1].total_return_pct)
        worst_performer = min(results.items(), key=lambda x: x[1].total_return_pct)
        
        # Correlation analysis (if multiple symbols)
        correlation_matrix = PortfolioAnalyzer._calculate_correlation_matrix(results)
        
        # Diversification metrics
        diversification_stats = PortfolioAnalyzer._calculate_diversification_stats(results)
        
        return {
            'portfolio_summary': {
                'total_symbols': len(results),
                'total_trades': total_trades,
                'total_profit': total_profit,
                'total_return_pct': total_return_pct,
                'avg_return_pct': total_return_pct / len(results),
                'weighted_win_rate': weighted_win_rate,
                'weighted_sharpe_ratio': weighted_sharpe,
                'weighted_max_drawdown': weighted_max_dd
            },
            'performance_ranking': [
                {
                    'symbol': symbol,
                    'return_pct': result.total_return_pct,
                    'trades': result.total_trades,
                    'win_rate': result.win_rate,
                    'sharpe_ratio': result.sharpe_ratio
                }
                for symbol, result in sorted(results.items(), 
                                           key=lambda x: x[1].total_return_pct, 
                                           reverse=True)
            ],
            'best_performer': {
                'symbol': best_performer[0],
                'return_pct': best_performer[1].total_return_pct,
                'trades': best_performer[1].total_trades
            },
            'worst_performer': {
                'symbol': worst_performer[0],
                'return_pct': worst_performer[1].total_return_pct,
                'trades': worst_performer[1].total_trades
            },
            'correlation_analysis': correlation_matrix,
            'diversification_stats': diversification_stats
        }
    
    @staticmethod
    def _empty_portfolio_stats() -> Dict[str, Any]:
        """Return empty portfolio statistics"""
        return {
            'portfolio_summary': {
                'total_symbols': 0,
                'total_trades': 0,
                'total_profit': 0.0,
                'total_return_pct': 0.0,
                'avg_return_pct': 0.0,
                'weighted_win_rate': 0.0,
                'weighted_sharpe_ratio': 0.0,
                'weighted_max_drawdown': 0.0
            },
            'performance_ranking': [],
            'best_performer': None,
            'worst_performer': None,
            'correlation_analysis': {},
            'diversification_stats': {}
        }
    
    @staticmethod
    def _calculate_correlation_matrix(results: Dict[str, BacktestResult]) -> Dict[str, Any]:
        """Calculate correlation matrix between symbol returns"""
        if len(results) < 2:
            return {}
        
        # Extract daily returns (simplified - using trade returns)
        symbol_returns = {}
        for symbol, result in results.items():
            if result.trades:
                returns = [trade.return_percentage / 100.0 for trade in result.trades]
                symbol_returns[symbol] = returns
        
        if len(symbol_returns) < 2:
            return {}
        
        # Calculate correlations (simplified)
        correlations = {}
        symbols = list(symbol_returns.keys())
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                returns1 = symbol_returns[symbol1]
                returns2 = symbol_returns[symbol2]
                
                # Align lengths
                min_len = min(len(returns1), len(returns2))
                if min_len > 1:
                    corr = np.corrcoef(returns1[:min_len], returns2[:min_len])[0, 1]
                    correlations[f"{symbol1}_{symbol2}"] = corr if not np.isnan(corr) else 0.0
        
        return correlations
    
    @staticmethod
    def _calculate_diversification_stats(results: Dict[str, BacktestResult]) -> Dict[str, Any]:
        """Calculate portfolio diversification statistics"""
        if len(results) < 2:
            return {}
        
        returns = [result.total_return_pct for result in results.values()]
        return {
            'return_std': np.std(returns, ddof=1) if len(returns) > 1 else 0.0,
            'return_range': max(returns) - min(returns),
            'coefficient_of_variation': np.std(returns, ddof=1) / np.mean(returns) if np.mean(returns) != 0 else 0.0
        }


class ReportGenerator:
    """Generate comprehensive analysis reports"""
    
    @staticmethod
    def generate_full_report(results: Dict[str, BacktestResult], 
                           strategy_name: str = "Strategy",
                           save_to_file: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report
        
        Args:
            results: Backtest results
            strategy_name: Name of the strategy
            save_to_file: Whether to save report to file
            
        Returns:
            Complete analysis report
        """
        report = {
            'metadata': {
                'strategy_name': strategy_name,
                'analysis_date': datetime.now().isoformat(),
                'symbols_analyzed': list(results.keys()),
                'total_symbols': len(results)
            },
            'portfolio_analysis': PortfolioAnalyzer.analyze_portfolio(results),
            'individual_symbol_analysis': {}
        }
        
        # Analyze each symbol individually
        for symbol, result in results.items():
            symbol_analysis = {
                'basic_stats': {
                    'total_trades': result.total_trades,
                    'winning_trades': result.winning_trades,
                    'losing_trades': result.losing_trades,
                    'win_rate': result.win_rate,
                    'total_return_pct': result.total_return_pct,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'profit_factor': result.profit_factor
                },
                'detailed_trade_analysis': TradeAnalyzer.analyze_trades(result.trades)
            }
            report['individual_symbol_analysis'][symbol] = symbol_analysis
        
        # Add summary insights
        report['insights'] = ReportGenerator._generate_insights(report)
        
        # Save to file if requested
        if save_to_file:
            ReportGenerator._save_report(report, strategy_name)
        
        return report
    
    @staticmethod
    def _generate_insights(report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable insights from the analysis"""
        insights = {
            'performance_insights': [],
            'risk_insights': [],
            'optimization_suggestions': []
        }
        
        portfolio = report['portfolio_analysis']['portfolio_summary']
        
        # Performance insights
        if portfolio['weighted_win_rate'] > 0.6:
            insights['performance_insights'].append("Strong win rate indicates good entry signals")
        elif portfolio['weighted_win_rate'] < 0.4:
            insights['performance_insights'].append("Low win rate suggests entry criteria may be too loose")
        
        if portfolio['weighted_sharpe_ratio'] > 1.5:
            insights['performance_insights'].append("Excellent risk-adjusted returns")
        elif portfolio['weighted_sharpe_ratio'] < 0.5:
            insights['performance_insights'].append("Poor risk-adjusted returns - consider risk management")
        
        # Risk insights
        if portfolio['weighted_max_drawdown'] > 20:
            insights['risk_insights'].append("High drawdown - consider position sizing or stop losses")
        
        # Optimization suggestions
        best_performer = report['portfolio_analysis'].get('best_performer')
        if best_performer and best_performer['return_pct'] > 5:
            insights['optimization_suggestions'].append(
                f"Focus optimization on {best_performer['symbol']} - showing strong performance"
            )
        
        return insights
    
    @staticmethod
    def _save_report(report: Dict[str, Any], strategy_name: str) -> Path:
        """Save report to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_report_{strategy_name}_{timestamp}.json"
        
        reports_dir = Path("analysis_reports")
        reports_dir.mkdir(exist_ok=True)
        
        filepath = reports_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Analysis report saved to {filepath}")
        return filepath
    
    @staticmethod
    def print_summary(results: Dict[str, BacktestResult], strategy_name: str = "Strategy"):
        """Print a quick summary to console"""
        portfolio_stats = PortfolioAnalyzer.analyze_portfolio(results)
        
        logger.info(f"\n📊 {strategy_name} ANALYSIS SUMMARY")
        logger.info("=" * 50)
        
        summary = portfolio_stats['portfolio_summary']
        logger.info(f"Symbols Tested: {summary['total_symbols']}")
        logger.info(f"Total Trades: {summary['total_trades']}")
        logger.info(f"Average Return: {summary['avg_return_pct']:.2f}%")
        logger.info(f"Win Rate: {summary['weighted_win_rate']:.1%}")
        logger.info(f"Sharpe Ratio: {summary['weighted_sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {summary['weighted_max_drawdown']:.2f}%")
        
        if portfolio_stats['best_performer']:
            best = portfolio_stats['best_performer']
            logger.info(f"\n🏆 Best Performer: {best['symbol']} ({best['return_pct']:.2f}%)")
        
        if portfolio_stats['worst_performer']:
            worst = portfolio_stats['worst_performer']
            logger.info(f"📉 Worst Performer: {worst['symbol']} ({worst['return_pct']:.2f}%)")


# Convenience functions
def analyze_backtest_results(results: Dict[str, BacktestResult], 
                           strategy_name: str = "Strategy") -> Dict[str, Any]:
    """
    Main function to analyze backtest results
    
    Args:
        results: Dictionary of symbol -> BacktestResult
        strategy_name: Name of strategy for reporting
        
    Returns:
        Complete analysis report
    """
    return ReportGenerator.generate_full_report(results, strategy_name)


def quick_analysis(results: Dict[str, BacktestResult], strategy_name: str = "Strategy"):
    """Quick analysis with console output"""
    ReportGenerator.print_summary(results, strategy_name)
    return ReportGenerator.generate_full_report(results, strategy_name, save_to_file=False)