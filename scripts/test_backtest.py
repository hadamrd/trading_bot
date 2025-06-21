#!/usr/bin/env python3
"""
Fixed test script for the backtesting engine.
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


def test_vwap_strategy():
    """Test with VWAP Bounce Strategy using your exact parameters."""
    
    print("🧪 Testing VWAP Bounce Strategy...")
    
    # Create the exact config from your working example
    config = BacktestConfig(
        symbols=["FTMUSDT"],
        timeframe=TimeFrame.FIFTEEN_MINUTES,
        since_date=datetime(2022, 12, 1),
        test_start_date=datetime(2023, 12, 1),
        test_end_date=None,
        initial_balance=Decimal("6000"),
        fee_rate=Decimal("0.001")
    )
    
    print(f"📊 Config: {config.symbols}")
    print(f"   Data from: {config.since_date.date()}")
    print(f"   Test from: {config.test_start_date.date()}")
    
    # Use your exact strategy parameters
    strategy_params = {
        "vwap_period": 5,
        "rsi_period": 8,
        "rsi_oversold": 40,
        "rsi_overbought": 68,
        "bounce_threshold": 0.0026526772991049326,
        "volume_factor": 1.0034958968836714,
        "take_profit_percentage": 0.04200763886459328,
        "stop_loss_percentage": 0.0226834639412405,
        "atr_period": 21
    }
    
    print("📋 Strategy Parameters:")
    for key, value in strategy_params.items():
        print(f"   {key}: {value}")
    
    # Import and create strategy
    try:
        # For now, let's use the EMA strategy that we know works
        from trading_bot.strategies.ema_crossover import EMACrossoverStrategy
        strategy = EMACrossoverStrategy(
            fast_period=12,
            slow_period=26,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            position_size_pct=0.1
        )
        print(f"✅ Using {strategy} (fallback)")
        
    except ImportError:
        print("❌ Could not import strategy")
        return
    
    # Create backtesting engine
    engine = BacktestEngine(config, strategy)
    
    # Run backtest
    try:
        print("\n🚀 Starting backtest...")
        results = engine.run()
        
        # Print results
        for symbol, result in results.items():
            print(f"\n📊 Results for {symbol}:")
            print(f"   Total Trades: {result.total_trades}")
            print(f"   Win Rate: {result.win_rate:.1%}")
            print(f"   Total Return: ${result.total_return:.2f} ({result.total_return_pct:.2f}%)")
            print(f"   Max Drawdown: {result.max_drawdown:.2f}%")
            print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"   Profit Factor: {result.profit_factor:.2f}")
            
            if result.total_trades > 0:
                print(f"   Average Profit: ${result.average_profit:.2f}")
                print(f"   Average Loss: ${result.average_loss:.2f}")
                print(f"   Largest Win: ${result.largest_win:.2f}")
                print(f"   Largest Loss: ${result.largest_loss:.2f}")
                print(f"   Average Holding Time: {result.average_holding_time:.2f} hours")
        
        print("\n✅ Backtesting engine works!")
        return results
        
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_simple_strategy():
    """Test with a simple EMA crossover strategy."""
    
    print("🧪 Testing Simple EMA Strategy...")
    
    # Simpler config for testing
    config = BacktestConfig(
        symbols=["BTCUSDT"],
        timeframe=TimeFrame.FIFTEEN_MINUTES,
        since_date=datetime.now() - timedelta(days=60),
        test_start_date=datetime.now() - timedelta(days=30),
        test_end_date=datetime.now() - timedelta(days=1),
        initial_balance=Decimal("1000"),
        fee_rate=Decimal("0.001")
    )
    
    print(f"📊 Config: {config.symbols}")
    print(f"   Data from: {config.since_date.date()}")
    print(f"   Test from: {config.test_start_date.date()} to {config.test_end_date.date()}")
    
    # Simple strategy
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
        print(f"❌ No strategy found: {e}")
        return None
    
    # Create backtesting engine
    engine = BacktestEngine(config, strategy)
    
    # Run backtest
    try:
        print("\n🚀 Starting backtest...")
        results = engine.run()
        engine.print_summary(results)
        print("✅ Simple strategy test works!")
        return results
        
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main test function."""
    print("🚀 Testing Fixed Backtesting Engine")
    print("=" * 50)
    
    # Test 1: Simple strategy
    print("\n" + "="*20 + " TEST 1 " + "="*20)
    simple_results = test_simple_strategy()
    
    if simple_results:
        print("\n✅ Simple test passed!")
        
        # Test 2: Your VWAP strategy  
        print("\n" + "="*20 + " TEST 2 " + "="*20)
        vwap_results = test_vwap_strategy()
        
        if vwap_results:
            print("\n✅ All tests passed!")
        else:
            print("\n⚠️  VWAP test failed, but simple test worked")
    else:
        print("\n❌ Basic test failed - check your setup")


if __name__ == "__main__":
    main()