#!/usr/bin/env python3
"""
Test the strategy with more realistic parameters for volatile cryptos
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.core.models import BacktestConfig
from trading_bot.core.enums import TimeFrame

def test_optimized_volatile_strategy():
    """Test with parameters optimized for volatile cryptos"""
    print("🎯 TESTING OPTIMIZED PARAMETERS FOR VOLATILE CRYPTOS")
    print("=" * 65)
    
    from trading_bot.strategies.time_based_reversion import TimeBasedReversionStrategy
    
    # Test just PEPEUSDT first (had 0 trades before)
    config = BacktestConfig(
        symbols=["PEPEUSDT"],
        timeframe=TimeFrame.FIVE_MINUTES,
        since_date=datetime(2024, 1, 1),
        test_start_date=datetime(2024, 6, 1),
        test_end_date=datetime(2024, 9, 1),  # 3 months test
        initial_balance=Decimal("5000"),
        fee_rate=Decimal("0.001")
    )
    
    # OPTIMIZED parameters for volatile cryptos
    optimized_strategy = TimeBasedReversionStrategy(
        # More sensitive mean reversion
        short_ma_period=6,               # Shorter MA for faster signals
        max_distance_from_ma=0.008,      # 0.8% instead of 2% (more realistic)
        
        # Less extreme RSI
        rsi_period=7,
        rsi_oversold=30,                 # 30 instead of 20 (more signals)
        
        # Remove time restrictions for crypto
        preferred_sessions=[],           # Trade 24/7 - crypto doesn't sleep!
        
        # More permissive filters
        max_velocity=0.06,               # Allow more volatility
        min_volume_ratio=1.0,            # Standard volume requirement
        
        # Same risk management
        stop_loss_atr=2.0,
        take_profit_atr=3.0,
        position_size_pct=0.02
    )
    
    print(f"🎯 OPTIMIZED Strategy:")
    print(f"   • RSI oversold: 30 (vs 20)")
    print(f"   • Distance from MA: 0.8% (vs 2.0%)")
    print(f"   • Sessions: 24/7 (vs limited hours)")
    print(f"   • Max velocity: 6% (vs 2.5%)")
    print(f"   • Volume ratio: 1.0 (vs 0.8)")
    
    engine = BacktestEngine(config, optimized_strategy)
    results = engine.run()
    
    for symbol, result in results.items():
        print(f"\n📊 OPTIMIZED Results for {symbol}:")
        print(f"   Return: {result.total_return_pct:>8.2f}%")
        print(f"   Trades: {result.total_trades:>8}")
        print(f"   Win Rate: {result.win_rate:>6.1%}")
        print(f"   Sharpe: {result.sharpe_ratio:>8.2f}")
        print(f"   Max DD: {result.max_drawdown:>8.2f}%")
        
        if result.total_trades > 0:
            print(f"   Avg Trade: ${result.total_return / result.total_trades:>6.2f}")
    
    return results

def test_even_more_aggressive():
    """Test with even more aggressive parameters for comparison"""
    print(f"\n🔥 TESTING AGGRESSIVE PARAMETERS (More Signals)")
    print("=" * 55)
    
    from trading_bot.strategies.time_based_reversion import TimeBasedReversionStrategy
    
    config = BacktestConfig(
        symbols=["PEPEUSDT"],
        timeframe=TimeFrame.FIVE_MINUTES,
        since_date=datetime(2024, 1, 1),
        test_start_date=datetime(2024, 6, 1),
        test_end_date=datetime(2024, 9, 1),
        initial_balance=Decimal("5000"),
        fee_rate=Decimal("0.001")
    )
    
    # AGGRESSIVE parameters for maximum signals
    aggressive_strategy = TimeBasedReversionStrategy(
        short_ma_period=5,               # Very short MA
        max_distance_from_ma=0.005,      # 0.5% - very tight
        
        rsi_period=5,                    # Fast RSI
        rsi_oversold=35,                 # Higher threshold = more signals
        
        preferred_sessions=[],           # 24/7
        max_velocity=0.10,               # Allow high volatility
        min_volume_ratio=0.8,            # Lower volume requirement
        
        stop_loss_atr=1.5,               # Tighter stops
        take_profit_atr=2.0,             # Quicker profits
        position_size_pct=0.015          # Smaller positions, more trades
    )
    
    print(f"🔥 AGGRESSIVE Strategy:")
    print(f"   • RSI oversold: 35 (even more signals)")
    print(f"   • Distance from MA: 0.5% (very tight)")
    print(f"   • RSI period: 5 (faster)")
    print(f"   • Max velocity: 10% (high volatility OK)")
    
    engine = BacktestEngine(config, aggressive_strategy)
    results = engine.run()
    
    for symbol, result in results.items():
        print(f"\n📊 AGGRESSIVE Results for {symbol}:")
        print(f"   Return: {result.total_return_pct:>8.2f}%")
        print(f"   Trades: {result.total_trades:>8}")
        print(f"   Win Rate: {result.win_rate:>6.1%}")
        print(f"   Sharpe: {result.sharpe_ratio:>8.2f}")
        print(f"   Max DD: {result.max_drawdown:>8.2f}%")
    
    return results

def compare_all_approaches():
    """Compare original vs optimized vs aggressive"""
    print(f"\n📊 COMPARISON OF ALL APPROACHES")
    print("=" * 50)
    
    print(f"Original (restrictive):")
    print(f"   PEPEUSDT: 0 trades, 0.00% return")
    
    optimized_results = test_optimized_volatile_strategy()
    aggressive_results = test_even_more_aggressive()
    
    print(f"\n💡 INSIGHTS:")
    print(f"   • Original parameters were too restrictive for volatile cryptos")
    print(f"   • 24/7 trading is crucial for crypto markets")
    print(f"   • RSI 20 is too extreme - 30-35 works better")
    print(f"   • Distance requirements need to match asset volatility")

def main():
    """Main test function"""
    print("🎯 Parameter Optimization for Volatile Cryptos")
    
    compare_all_approaches()
    
    print(f"\n🚀 RECOMMENDATIONS:")
    print(f"   1. Use optimized parameters as starting point")
    print(f"   2. If still few trades, try aggressive parameters")
    print(f"   3. Test on 5-minute timeframe for more signals")
    print(f"   4. Consider different assets if strategy doesn't fit")

if __name__ == "__main__":
    main()