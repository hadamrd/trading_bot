#!/usr/bin/env python3
"""
Test optimized Time-Based Reversion on Jan-May 2025 data
Simple test, no fluff
"""

import sys
from pathlib import Path
from datetime import datetime
from decimal import Decimal

sys.path.append(str(Path(__file__).parent.parent / "src"))

from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.strategies.time_based_reversion import TimeBasedReversionStrategy
from trading_bot.core.models import BacktestConfig
from trading_bot.core.enums import TimeFrame


def test_time_based_2025():
    """Test on 5 months of 2025 data"""
    
    print("Testing Time-Based Strategy: Jan-May 2025")
    print("=" * 50)
    
    config = BacktestConfig(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "PEPEUSDT"],
        timeframe=TimeFrame.FIVE_MINUTES,
        since_date=datetime(2024, 12, 1),   # Extra data for indicators
        test_start_date=datetime(2025, 1, 1),
        test_end_date=datetime(2025, 5, 31),
        initial_balance=Decimal("10000"),
        fee_rate=Decimal("0.001")
    )
    
    print(f"Period: {config.test_start_date.date()} to {config.test_end_date.date()}")
    print(f"Symbols: {', '.join(config.symbols)}")
    
    # Default parameters
    default_strategy = TimeBasedReversionStrategy(
        short_ma_period=10,
        max_distance_from_ma=0.008,
        rsi_period=9,
        rsi_oversold=30,
        max_velocity=0.025,
        min_volume_ratio=0.8,
        stop_loss_atr=2.0,
        take_profit_atr=3.0,
        position_size_pct=0.025
    )
    
    # Optimized parameters
    optimized_strategy = TimeBasedReversionStrategy(
        short_ma_period=9,
        max_distance_from_ma=0.006651955567737569,
        rsi_period=12,
        rsi_oversold=28,
        max_velocity=0.04860951682928848,
        min_volume_ratio=0.807699640691528,
        stop_loss_atr=2.365189676118769,
        take_profit_atr=2.747075989408798,
        position_size_pct=0.03960476471461546
    )
    
    strategies = {
        "Default": default_strategy,
        "Optimized": optimized_strategy
    }
    
    results = {}
    
    for name, strategy in strategies.items():
        print(f"\nTesting {name}...")
        
        engine = BacktestEngine(config, strategy)
        strategy_results = engine.run()
        results[name] = strategy_results
        
        # Portfolio totals
        total_return = sum(r.total_return_pct for r in strategy_results.values())
        total_trades = sum(r.total_trades for r in strategy_results.values())
        avg_win_rate = sum(r.win_rate for r in strategy_results.values()) / len(strategy_results)
        
        print(f"Total Return: {total_return:>8.2f}%")
        print(f"Total Trades: {total_trades:>8}")
        print(f"Avg Win Rate: {avg_win_rate:>8.1%}")
    
    # Results table
    print(f"\nDetailed Results:")
    print(f"{'Symbol':<10} {'Strategy':<10} {'Return':<8} {'Trades':<7} {'Win%':<6}")
    print("-" * 50)
    
    for symbol in config.symbols:
        for strategy_name in strategies.keys():
            result = results[strategy_name][symbol]
            print(f"{symbol:<10} {strategy_name:<10} {result.total_return_pct:>6.2f}%  {result.total_trades:>5}   {result.win_rate:>5.1%}")
    
    # Summary
    default_total = sum(r.total_return_pct for r in results["Default"].values())
    optimized_total = sum(r.total_return_pct for r in results["Optimized"].values())
    improvement = optimized_total - default_total
    
    print(f"\nSummary:")
    print(f"Default Total:    {default_total:>6.2f}%")
    print(f"Optimized Total:  {optimized_total:>6.2f}%")
    print(f"Difference:       {improvement:>+6.2f}%")
    
    if optimized_total > 2.0:
        print("✅ Strategy is profitable")
    elif optimized_total > 0:
        print("⚖️ Strategy barely profitable")
    else:
        print("❌ Strategy loses money")
    
    return results


if __name__ == "__main__":
    test_time_based_2025()