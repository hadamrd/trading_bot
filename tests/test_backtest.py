#!/usr/bin/env python3
"""
Test script for research-backed trading strategies.
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


def test_multi_regime_strategy():
    """Test the Multi-Regime strategy."""
    print("🧪 Testing Multi-Regime Strategy (Volatility-Based)")
    
    config = BacktestConfig(
        symbols=["BTCUSDT"],
        timeframe=TimeFrame.FIFTEEN_MINUTES,
        since_date=datetime(2024, 1, 1),
        test_start_date=datetime(2024, 6, 1),
        test_end_date=datetime(2024, 12, 1),
        initial_balance=Decimal("5000"),
        fee_rate=Decimal("0.001")
    )
    
    # Save this as: src/trading_bot/strategies/multi_regime.py
    from trading_bot.strategies.multi_regime import MultiRegimeStrategy
    
    strategy = MultiRegimeStrategy(
        # Regime detection
        volatility_threshold_high=0.7,
        volatility_threshold_low=0.3,
        
        # Mean reversion in low vol
        rsi_oversold=30,
        bb_std=2.0,
        
        # Momentum in high vol
        ema_fast=9,
        ema_slow=21,
        
        # Risk management
        stop_loss_atr=2.0,
        take_profit_atr=3.0,
        position_size_pct=0.02
    )
    
    engine = BacktestEngine(config, strategy)
    results = engine.run()
    
    for symbol, result in results.items():
        print(f"\n📊 Multi-Regime Results for {symbol}:")
        print(f"   Trades: {result.total_trades}")
        print(f"   Win Rate: {result.win_rate:.1%}")
        print(f"   Return: {result.total_return_pct:.2f}%")
        print(f"   Sharpe: {result.sharpe_ratio:.2f}")
        print(f"   Max DD: {result.max_drawdown:.2f}%")
    
    return results


def test_vwap_statistical_strategy():
    """Test the VWAP Statistical strategy."""
    print("\n🧪 Testing VWAP Statistical Strategy")
    
    config = BacktestConfig(
        symbols=["ETHUSDT"],
        timeframe=TimeFrame.FIFTEEN_MINUTES,
        since_date=datetime(2024, 1, 1),
        test_start_date=datetime(2024, 6, 1),
        test_end_date=datetime(2024, 12, 1),
        initial_balance=Decimal("5000"),
        fee_rate=Decimal("0.001")
    )
    
    # Save this as: src/trading_bot/strategies/vwap_statistical.py
    from trading_bot.strategies.vwap_statistical import VWAPStatisticalStrategy
    
    strategy = VWAPStatisticalStrategy(
        # VWAP parameters
        vwap_period=20,
        zscore_period=20,
        
        # Entry/exit thresholds
        zscore_entry_threshold=-1.5,  # Strong deviation
        zscore_exit_threshold=0.5,
        
        # Confirmation filters
        min_volume_ratio=1.2,
        rsi_oversold=35,
        max_vwap_slope=0.002,
        
        # Risk management
        stop_loss_atr=2.5,
        take_profit_multiple=2.0,
        position_size_pct=0.03
    )
    
    engine = BacktestEngine(config, strategy)
    results = engine.run()
    
    for symbol, result in results.items():
        print(f"\n📊 VWAP Statistical Results for {symbol}:")
        print(f"   Trades: {result.total_trades}")
        print(f"   Win Rate: {result.win_rate:.1%}")
        print(f"   Return: {result.total_return_pct:.2f}%")
        print(f"   Sharpe: {result.sharpe_ratio:.2f}")
        print(f"   Max DD: {result.max_drawdown:.2f}%")
    
    return results


def test_time_based_strategy():
    """Test the Time-Based strategy."""
    print("\n🧪 Testing Time-Based Reversion Strategy")
    
    config = BacktestConfig(
        symbols=["ADAUSDT"],
        timeframe=TimeFrame.FIFTEEN_MINUTES,
        since_date=datetime(2024, 1, 1),
        test_start_date=datetime(2024, 6, 1),
        test_end_date=datetime(2024, 12, 1),
        initial_balance=Decimal("5000"),
        fee_rate=Decimal("0.001")
    )
    
    # Save this as: src/trading_bot/strategies/time_based_reversion.py
    from trading_bot.strategies.time_based_reversion import TimeBasedReversionStrategy
    
    strategy = TimeBasedReversionStrategy(
        # Mean reversion
        short_ma_period=10,
        max_distance_from_ma=0.008,
        
        # RSI
        rsi_period=9,
        rsi_oversold=25,
        
        # Time filters
        preferred_sessions=['european_morning', 'us_morning'],
        
        # Risk management
        stop_loss_atr=1.5,
        take_profit_atr=2.5,
        position_size_pct=0.025
    )
    
    engine = BacktestEngine(config, strategy)
    results = engine.run()
    
    for symbol, result in results.items():
        print(f"\n📊 Time-Based Results for {symbol}:")
        print(f"   Trades: {result.total_trades}")
        print(f"   Win Rate: {result.win_rate:.1%}")
        print(f"   Return: {result.total_return_pct:.2f}%")
        print(f"   Sharpe: {result.sharpe_ratio:.2f}")
        print(f"   Max DD: {result.max_drawdown:.2f}%")
    
    return results


def compare_strategies():
    """Compare all three research-backed strategies."""
    print("🚀 Testing Research-Backed Strategies")
    print("=" * 60)
    
    results = {}
    
    try:
        results['multi_regime'] = test_multi_regime_strategy()
    except Exception as e:
        print(f"❌ Multi-Regime failed: {e}")
    
    try:
        results['vwap_statistical'] = test_vwap_statistical_strategy()
    except Exception as e:
        print(f"❌ VWAP Statistical failed: {e}")
    
    try:
        results['time_based'] = test_time_based_strategy()
    except Exception as e:
        print(f"❌ Time-Based failed: {e}")
    
    # Summary comparison
    print(f"\n📊 STRATEGY COMPARISON")
    print(f"=" * 50)
    
    for strategy_name, strategy_results in results.items():
        if strategy_results:
            symbol = list(strategy_results.keys())[0]
            result = strategy_results[symbol]
            
            print(f"\n{strategy_name.replace('_', ' ').title()}:")
            print(f"  Return: {result.total_return_pct:>6.2f}%")
            print(f"  Trades: {result.total_trades:>6}")
            print(f"  Win%:   {result.win_rate:>6.1%}")
            print(f"  Sharpe: {result.sharpe_ratio:>6.2f}")
    
    print(f"\n💡 Strategy Recommendations:")
    print(f"   Multi-Regime: Best for volatile/changing markets")
    print(f"   VWAP Statistical: Best for range-bound markets")
    print(f"   Time-Based: Best for consistent daily patterns")


def main():
    """Main test function."""
    print("🎯 Testing Solid, Research-Backed Strategies")
    print(f"Based on academic research and market microstructure principles")
    print("=" * 70)
    
    compare_strategies()
    
    print(f"\n✅ Testing completed!")
    print(f"\n📚 Strategy Sources:")
    print(f"   • Mean reversion research in crypto markets")
    print(f"   • Market microstructure and order flow analysis")
    print(f"   • Time-of-day effects in cryptocurrency trading")
    print(f"   • Multi-regime volatility modeling")


if __name__ == "__main__":
    main()