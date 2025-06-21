#!/usr/bin/env python3
"""
Test script with automatic data downloading.
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


def test_auto_download():
    """Test with automatic data downloading."""
    
    print("🧪 Testing Backtest with Auto Data Download")
    
    # Test config with symbols that might not have data
    config = BacktestConfig(
        symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT"],  # Multiple symbols
        timeframe=TimeFrame.FIFTEEN_MINUTES,
        since_date=datetime(2024, 1, 1),  # 1 year of data
        test_start_date=datetime(2024, 6, 1),  # 6 months of backtest
        test_end_date=datetime(2024, 12, 1),
        initial_balance=Decimal("1000"),
        fee_rate=Decimal("0.001")
    )
    
    print(f"📊 Config:")
    print(f"   Symbols: {config.symbols}")
    print(f"   Data needed from: {config.since_date.date()}")
    print(f"   Backtest period: {config.test_start_date.date()} to {config.test_end_date.date()}")
    
    # Create strategy
    try:
        from trading_bot.strategies.ema_crossover import EMACrossoverStrategy
        strategy = EMACrossoverStrategy(
            fast_period=12,
            slow_period=26,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            position_size_pct=0.1
        )
        print(f"✅ Using {strategy}")
    except ImportError as e:
        print(f"❌ Strategy import failed: {e}")
        return
    
    # Create backtesting engine
    engine = BacktestEngine(config, strategy)
    
    # Optional: Check data availability first (doesn't download)
    print(f"\n🔍 Checking data availability...")
    data_available = engine.validate_data_availability()
    
    if not data_available:
        print(f"💾 Some data is missing, but backtest will auto-download as needed")
    
    # Run backtest (will auto-download missing data)
    try:
        results = engine.run()
        
        # Print summary
        print(f"\n📊 BACKTEST RESULTS")
        print(f"=" * 50)
        
        for symbol, result in results.items():
            print(f"\n{symbol}:")
            print(f"  Trades: {result.total_trades}")
            print(f"  Win Rate: {result.win_rate:.1%}")
            print(f"  Return: {result.total_return_pct:.2f}%")
            print(f"  Sharpe: {result.sharpe_ratio:.2f}")
        
        print("\n✅ Auto-download backtest completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_new_symbol():
    """Test with a symbol that definitely doesn't have data."""
    
    print("\n🧪 Testing with new symbol (DOGEUSDT)")
    
    config = BacktestConfig(
        symbols=["DOGEUSDT"],  # Probably don't have this data
        timeframe=TimeFrame.FIFTEEN_MINUTES,
        since_date=datetime.now() - timedelta(days=60),
        test_start_date=datetime.now() - timedelta(days=30),
        test_end_date=datetime.now() - timedelta(days=1),
        initial_balance=Decimal("1000"),
        fee_rate=Decimal("0.001")
    )
    
    from trading_bot.strategies.ema_crossover import EMACrossoverStrategy
    strategy = EMACrossoverStrategy()
    
    engine = BacktestEngine(config, strategy)
    
    try:
        results = engine.run()
        
        for symbol, result in results.items():
            print(f"\n{symbol}: {result.total_trades} trades, {result.total_return_pct:.2f}% return")
        
        print("✅ New symbol test passed!")
        return results
        
    except Exception as e:
        print(f"❌ New symbol test failed: {e}")
        return None


def main():
    """Main test function."""
    print("🚀 Testing Auto Data Download Feature")
    print("=" * 60)
    
    # Test 1: Multiple symbols with auto-download
    results1 = test_auto_download()
    
    if results1:
        # Test 2: New symbol
        results2 = test_new_symbol()
        
        if results2:
            print("\n🎉 All auto-download tests passed!")
        else:
            print("\n⚠️  Second test failed, but first passed")
    else:
        print("\n❌ Auto-download tests failed")


if __name__ == "__main__":
    main()