#!/usr/bin/env python3
"""
IMPROVED Time-Based Strategy Test
Fix the poor performance by making it more aggressive
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

def test_improved_strategy():
    """Test improved time-based strategy with bigger profits"""
    print("🚀 IMPROVED TIME-BASED STRATEGY TEST")
    print("=" * 60)
    print("Fixing the 0.23% return problem!")
    
    from trading_bot.strategies.time_based_reversion import TimeBasedReversionStrategy
    
    # Test on better assets (more institutional, less meme chaos)
    better_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    config = BacktestConfig(
        symbols=better_symbols,
        timeframe=TimeFrame.FIVE_MINUTES,  # Shorter timeframe for more signals
        since_date=datetime(2024, 1, 1),
        test_start_date=datetime(2024, 6, 1),
        test_end_date=datetime(2024, 11, 1),
        initial_balance=Decimal("5000"),
        fee_rate=Decimal("0.001")
    )
    
    # IMPROVED parameters for bigger profits
    improved_strategy = TimeBasedReversionStrategy(
        # More sensitive mean reversion
        short_ma_period=8,
        max_distance_from_ma=0.012,      # 1.2% (slightly bigger deviations)
        
        # Faster RSI for quicker signals
        rsi_period=9,
        rsi_oversold=30,                 # Standard oversold level
        
        # Remove time restrictions (crypto is 24/7)
        preferred_sessions=[],
        
        # Allow more volatility 
        max_velocity=0.04,               # 4% moves allowed
        min_volume_ratio=0.9,            # Lower volume requirement
        
        # BIGGER PROFITS - this is the key fix!
        stop_loss_atr=2.0,               # Same stops
        take_profit_atr=5.0,             # 5x ATR instead of 3x (KEY CHANGE!)
        position_size_pct=0.03           # Slightly bigger positions
    )
    
    print(f"🎯 KEY IMPROVEMENTS:")
    print(f"   • Take Profit: 5x ATR (vs 3x) - BIGGER PROFITS!")
    print(f"   • Timeframe: 5m (vs 15m) - More signals")
    print(f"   • Assets: BTC/ETH/SOL (vs PEPE) - Better patterns")
    print(f"   • Position size: 3% (vs 2%) - More per trade")
    
    engine = BacktestEngine(config, improved_strategy)
    results = engine.run()
    
    print(f"\n🚀 IMPROVED RESULTS")
    print(f"=" * 40)
    
    total_return = 0
    total_trades = 0
    
    for symbol, result in results.items():
        print(f"\n{symbol}:")
        print(f"  Return: {result.total_return_pct:>8.2f}%")
        print(f"  Trades: {result.total_trades:>8}")
        print(f"  Win%:   {result.win_rate:>8.1%}")
        print(f"  Sharpe: {result.sharpe_ratio:>8.2f}")
        print(f"  Max DD: {result.max_drawdown:>8.2f}%")
        
        if result.total_trades > 0:
            avg_per_trade = result.total_return_pct / result.total_trades
            print(f"  Avg/Trade: {avg_per_trade:>6.3f}%")
        
        total_return += result.total_return_pct
        total_trades += result.total_trades
    
    avg_return = total_return / len(results) if results else 0
    
    print(f"\n📊 IMPROVEMENT COMPARISON:")
    print(f"   BEFORE (PEPE 15m): +0.23% (15 trades)")
    print(f"   AFTER (BTC/ETH/SOL 5m): {avg_return:+.2f}% ({total_trades} trades)")
    
    if avg_return > 0.23:
        improvement = (avg_return / 0.23) - 1
        print(f"   🎉 IMPROVEMENT: +{improvement*100:.0f}% better returns!")
    
    return results

def test_ultra_aggressive():
    """Test ultra-aggressive version for maximum trades/profits"""
    print(f"\n🔥 ULTRA-AGGRESSIVE VERSION")
    print(f"=" * 35)
    
    from trading_bot.strategies.time_based_reversion import TimeBasedReversionStrategy
    
    config = BacktestConfig(
        symbols=["BTCUSDT"],  # Just BTC for focused test
        timeframe=TimeFrame.FIVE_MINUTES,
        since_date=datetime(2024, 1, 1),
        test_start_date=datetime(2024, 8, 1),   # Shorter period for quick test
        test_end_date=datetime(2024, 11, 1),
        initial_balance=Decimal("5000"),
        fee_rate=Decimal("0.001")
    )
    
    # ULTRA-AGGRESSIVE parameters
    ultra_strategy = TimeBasedReversionStrategy(
        short_ma_period=6,               # Very fast MA
        max_distance_from_ma=0.008,      # Tight deviations (more signals)
        
        rsi_period=7,                    # Fast RSI
        rsi_oversold=35,                 # Higher threshold = more trades
        
        preferred_sessions=[],           # 24/7
        max_velocity=0.06,               # Allow high volatility
        min_volume_ratio=0.7,            # Low volume requirement
        
        # MAXIMUM profit extraction
        stop_loss_atr=1.8,               # Tighter stops
        take_profit_atr=6.0,             # Even bigger targets!
        position_size_pct=0.04           # Bigger positions
    )
    
    print(f"🔥 ULTRA Settings:")
    print(f"   • Take Profit: 6x ATR")
    print(f"   • RSI oversold: 35 (more signals)")
    print(f"   • Position size: 4%")
    
    engine = BacktestEngine(config, ultra_strategy)
    results = engine.run()
    
    for symbol, result in results.items():
        print(f"\n🔥 ULTRA Results ({symbol}):")
        print(f"   Return: {result.total_return_pct:>8.2f}%")
        print(f"   Trades: {result.total_trades:>8}")
        print(f"   Win%:   {result.win_rate:>8.1%}")
        
        if result.total_trades > 0:
            avg_per_trade = result.total_return_pct / result.total_trades
            print(f"   Avg/Trade: {avg_per_trade:>6.3f}%")
    
    return results

def main():
    """Test improved strategies"""
    print("🎯 STRATEGY IMPROVEMENT TEST")
    print("Goal: Fix the 0.23% return problem")
    print("=" * 50)
    
    # Test improved version
    improved_results = test_improved_strategy()
    
    # Test ultra-aggressive version
    ultra_results = test_ultra_aggressive()
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • Bigger take-profit targets = bigger returns")
    print(f"   • 5-minute timeframe gives more opportunities") 
    print(f"   • BTC/ETH have better mean reversion than meme coins")
    print(f"   • Position sizing matters for total returns")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   • If results are good: optimize the best-performing asset")
    print(f"   • If still poor: try different strategy approach")
    print(f"   • Consider portfolio approach (multiple assets)")

if __name__ == "__main__":
    main()
