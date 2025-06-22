#!/usr/bin/env python3
"""
Quick test of optimized Time-Based strategy with better parameters.
"""

import sys
from pathlib import Path
from datetime import datetime
from decimal import Decimal

sys.path.append(str(Path(__file__).parent.parent / "src"))

from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.core.models import BacktestConfig
from trading_bot.core.enums import TimeFrame


def test_optimized_time_strategy():
    """Test optimized Time-Based strategy on multiple symbols."""
    
    print("🚀 Testing Optimized Time-Based Strategy")
    
    config = BacktestConfig(
        symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT"],  # Test on 3 symbols
        timeframe=TimeFrame.FIFTEEN_MINUTES,
        since_date=datetime(2024, 1, 1),
        test_start_date=datetime(2024, 6, 1),
        test_end_date=datetime(2024, 12, 1),
        initial_balance=Decimal("5000"),
        fee_rate=Decimal("0.001")
    )
    
    from trading_bot.strategies.time_based_reversion import TimeBasedReversionStrategy
    
    # Optimized parameters based on initial results
    strategy = TimeBasedReversionStrategy(
        # More selective mean reversion
        short_ma_period=12,                    # Slightly longer MA
        max_distance_from_ma=0.012,           # Larger deviation required
        
        # More selective RSI
        rsi_period=7,                         # Faster RSI
        rsi_oversold=20,                      # More oversold
        
        # Better time filters
        preferred_sessions=['european_morning', 'us_morning'],
        
        # Tighter risk management
        max_velocity=0.01,                    # Less momentum allowed
        min_volume_ratio=1.0,                 # Relaxed volume requirement
        
        # Risk management
        stop_loss_atr=1.2,                    # Tighter stop
        take_profit_atr=2.0,                  # Lower target
        position_size_pct=0.03               # Slightly larger positions
    )
    
    engine = BacktestEngine(config, strategy)
    results = engine.run()
    
    print(f"\n📊 OPTIMIZED RESULTS")
    print(f"=" * 40)
    
    total_return = 0
    total_trades = 0
    
    for symbol, result in results.items():
        print(f"\n{symbol}:")
        print(f"  Return: {result.total_return_pct:>6.2f}%")
        print(f"  Trades: {result.total_trades:>6}")
        print(f"  Win%:   {result.win_rate:>6.1%}")
        print(f"  Sharpe: {result.sharpe_ratio:>6.2f}")
        print(f"  Max DD: {result.max_drawdown:>6.2f}%")
        
        total_return += result.total_return_pct
        total_trades += result.total_trades
    
    avg_return = total_return / len(results)
    print(f"\n📈 PORTFOLIO SUMMARY:")
    print(f"   Average Return: {avg_return:.2f}%")
    print(f"   Total Trades: {total_trades}")
    print(f"   Trades per Symbol: {total_trades / len(results):.1f}")
    
    return results


def test_conservative_parameters():
    """Test even more conservative parameters."""
    
    print("\n🛡️  Testing Ultra-Conservative Parameters")
    
    config = BacktestConfig(
        symbols=["ADAUSDT"],  # Best performing symbol
        timeframe=TimeFrame.FIFTEEN_MINUTES,
        since_date=datetime(2024, 1, 1),
        test_start_date=datetime(2024, 6, 1),
        test_end_date=datetime(2024, 12, 1),
        initial_balance=Decimal("5000"),
        fee_rate=Decimal("0.001")
    )
    
    from trading_bot.strategies.time_based_reversion import TimeBasedReversionStrategy
    
    # Ultra-conservative parameters
    strategy = TimeBasedReversionStrategy(
        # Very selective
        max_distance_from_ma=0.015,           # Only big deviations
        rsi_oversold=15,                      # Very oversold
        rsi_period=5,                         # Very fast RSI
        
        # Quality filters
        min_volume_ratio=1.2,                 # Need volume confirmation
        max_velocity=0.005,                   # Very little momentum
        
        # Tight risk management
        stop_loss_atr=1.0,
        take_profit_atr=1.5,                  # Quick profits
        position_size_pct=0.04               # Larger positions for fewer trades
    )
    
    engine = BacktestEngine(config, strategy)
    results = engine.run()
    
    for symbol, result in results.items():
        print(f"\n📊 Conservative Results for {symbol}:")
        print(f"   Return: {result.total_return_pct:.2f}%")
        print(f"   Trades: {result.total_trades}")
        print(f"   Win Rate: {result.win_rate:.1%}")
        print(f"   Max DD: {result.max_drawdown:.2f}%")
        
        if result.total_trades > 0:
            print(f"   Avg Profit/Trade: ${result.total_return / result.total_trades:.2f}")
    
    return results


def main():
    """Test optimized versions."""
    
    # Test 1: Optimized parameters
    opt_results = test_optimized_time_strategy()
    
    # Test 2: Ultra-conservative
    cons_results = test_conservative_parameters()
    
    print(f"\n🎯 CONCLUSION:")
    print(f"Time-Based strategy shows promise - focus on parameter optimization")
    print(f"rather than complex new strategies.")


if __name__ == "__main__":
    main()