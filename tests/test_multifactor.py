#!/usr/bin/env python3
"""
Test Multi-Factor Strategy on Jan-May 2025 data
"""

import sys
from pathlib import Path
from datetime import datetime
from decimal import Decimal

sys.path.append(str(Path(__file__).parent.parent / "src"))

from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.strategies.multi_factor import MultiFactorStrategy
from trading_bot.core.models import BacktestConfig
from trading_bot.core.enums import TimeFrame


def test_multi_factor_2025():
    """Test Multi-Factor strategy configurations"""
    
    print("Multi-Factor Strategy Test: Jan-May 2025")
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
    
    # Test 3 configurations
    strategies = {
        "All_Signals": MultiFactorStrategy(),  # All 6 signals enabled
        
        "VWAP_Only": MultiFactorStrategy(
            use_short_cross=False,
            use_long_cross=False,
            use_short_bounce=False,
            use_long_bounce=False,
            use_vwap_bounce=True,
            use_ema_golden_cross=False
        ),
        
        "Best_3": MultiFactorStrategy(
            use_short_cross=True,
            use_long_cross=False,
            use_short_bounce=True,
            use_long_bounce=False,
            use_vwap_bounce=True,
            use_ema_golden_cross=True
        )
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
    
    # Detailed results
    print(f"\nDetailed Results:")
    print(f"{'Symbol':<10} {'All Signals':<12} {'VWAP Only':<12} {'Best 3':<12}")
    print("-" * 55)
    
    totals = {"All_Signals": 0, "VWAP_Only": 0, "Best_3": 0}
    
    for symbol in config.symbols:
        row_data = [symbol]
        for strategy_name in ["All_Signals", "VWAP_Only", "Best_3"]:
            ret = results[strategy_name][symbol].total_return_pct
            totals[strategy_name] += ret
            row_data.append(f"{ret:+6.2f}%")
        
        print(f"{row_data[0]:<10} {row_data[1]:<12} {row_data[2]:<12} {row_data[3]:<12}")
    
    print("-" * 55)
    print(f"{'TOTAL':<10} {totals['All_Signals']:>+6.2f}%      {totals['VWAP_Only']:>+6.2f}%      {totals['Best_3']:>+6.2f}%")
    
    # Find best configuration
    best_config = max(totals.keys(), key=lambda k: totals[k])
    best_return = totals[best_config]
    
    print(f"\nResults Summary:")
    if best_return > 3.0:
        print(f"✅ {best_config} is profitable: {best_return:+.2f}%")
        print("🎉 Finally found a working strategy!")
    elif best_return > 0:
        print(f"⚖️ {best_config} barely profitable: {best_return:+.2f}%")
        print("💡 Shows promise, might work with optimization")
    else:
        print(f"❌ All configurations lose money")
        print(f"Best was {best_config}: {best_return:+.2f}%")
    
    # Compare to previous strategies
    print(f"\nComparison to Previous Tests:")
    print(f"EMA Crossover:     -3.08%  ❌")
    print(f"Time-Based:        -0.82%  ❌") 
    print(f"Multi-Factor Best: {best_return:+6.2f}%  {'✅' if best_return > 0 else '❌'}")
    
    return results


if __name__ == "__main__":
    test_multi_factor_2025()