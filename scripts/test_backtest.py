#!/usr/bin/env python3
"""
Simple test script for the backtesting engine.
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


def main():
    """Test the backtesting engine with a simple strategy."""
    
    print("🧪 Testing Backtesting Engine...")
    
    # Create simple backtest config
    config = BacktestConfig(
        symbols=["BTCUSDT"],
        timeframe=TimeFrame.FIFTEEN_MINUTES,
        since_date=datetime.now() - timedelta(days=60),  # Data collection start
        test_start_date=datetime.now() - timedelta(days=30),  # Backtest start
        test_end_date=datetime.now() - timedelta(days=1),     # Backtest end
        initial_balance=Decimal("1000"),
        fee_rate=Decimal("0.001")
    )
    
    print(f"📊 Config: {config.symbols}")
    print(f"   Data from: {config.since_date.date()}")
    print(f"   Test from: {config.test_start_date.date()} to {config.test_end_date.date()}")
    
    # We need a strategy to test with
    try:
        from trading_bot.strategies.ema_crossover import EMACrossoverStrategy
        strategy = EMACrossoverStrategy(
            fast_period=12,
            slow_period=26,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            position_size_pct=0.1  # Use 10% of balance per trade
        )
        print(f"✅ Using {strategy}")
    except ImportError as e:
        print(f"❌ No strategy found: {e}")
        return
    
    # Create backtesting engine
    engine = BacktestEngine(config, strategy)
    
    # Run backtest
    try:
        results = engine.run()
        engine.print_summary(results)
        print("✅ Backtesting engine works!")
        
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()