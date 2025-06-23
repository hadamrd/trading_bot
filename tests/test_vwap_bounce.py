#!/usr/bin/env python3
"""
Test VWAP Bounce strategy vs Time-Based on Jan-May 2025
"""

import sys
from pathlib import Path
from datetime import datetime
from decimal import Decimal

sys.path.append(str(Path(__file__).parent.parent / "src"))

from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.strategies.vwap_bounce import VWAPBounceStrategy
from trading_bot.strategies.time_based_reversion import TimeBasedReversionStrategy
from trading_bot.core.models import BacktestConfig
from trading_bot.core.enums import TimeFrame


def test_vwap_bounce_2025():
    """Test VWAP Bounce vs Time-Based on 2025 data"""
    
    print("VWAP Bounce vs Time-Based: Jan-May 2025")
    print("=" * 50)
    
    config = BacktestConfig(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "PEPEUSDT"],
        timeframe=TimeFrame.FIVE_MINUTES,
        since_date=datetime(2024, 12, 1),
        test_start_date=datetime(2025, 1, 1),
        test_end_date=datetime(2025, 5, 31),
        initial_balance=Decimal("10000"),
        fee_rate=Decimal("0.001")
    )
    
    print(f"Period: {config.test_start_date.date()} to {config.test_end_date.date()}")
    print(f"Symbols: {', '.join(config.symbols)}")
    
    # VWAP Bounce strategy (using default parameters)
    vwap_strategy = VWAPBounceStrategy()
    
    # Time-Based (for comparison)
    time_strategy = TimeBasedReversionStrategy(
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
        "VWAP_Bounce": vwap_strategy,
        "Time_Based": time_strategy
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
    
    # Results comparison
    print(f"\nResults by Symbol:")
    print(f"{'Symbol':<10} {'VWAP Bounce':<12} {'Time-Based':<12} {'Difference':<12}")
    print("-" * 55)
    
    vwap_total = 0
    time_total = 0
    
    for symbol in config.symbols:
        vwap_return = results["VWAP_Bounce"][symbol].total_return_pct
        time_return = results["Time_Based"][symbol].total_return_pct
        diff = vwap_return - time_return
        
        vwap_total += vwap_return
        time_total += time_return
        
        print(f"{symbol:<10} {vwap_return:>+6.2f}%      {time_return:>+6.2f}%      {diff:>+6.2f}%")
    
    total_diff = vwap_total - time_total
    
    print("-" * 55)
    print(f"{'TOTAL':<10} {vwap_total:>+6.2f}%      {time_total:>+6.2f}%      {total_diff:>+6.2f}%")
    
    # Verdict
    print(f"\nVerdict:")
    if vwap_total > 2.0:
        print("✅ VWAP Bounce is profitable")
    elif vwap_total > 0:
        print("⚖️ VWAP Bounce barely profitable")
    else:
        print("❌ VWAP Bounce also loses money")
    
    if total_diff > 1.0:
        print("🎯 VWAP Bounce significantly better than Time-Based")
    elif total_diff > 0:
        print("📈 VWAP Bounce slightly better than Time-Based")
    else:
        print("📉 VWAP Bounce worse than Time-Based")
    
    return results


if __name__ == "__main__":
    test_vwap_bounce_2025()