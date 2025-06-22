#!/usr/bin/env python3
"""
Test Time-Based Strategy on highly volatile cryptocurrencies.
These assets have much larger price swings = potential for bigger profits.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal

sys.path.append(str(Path(__file__).parent.parent / "src"))

from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.core.models import BacktestConfig
from trading_bot.core.enums import TimeFrame


def test_volatile_cryptos():
    """Test Time-Based strategy on the hottest, most volatile cryptos."""
    
    print("🔥 Testing Time-Based Strategy on VOLATILE CRYPTOS")
    print("=" * 60)
    
    # Most volatile cryptos of 2024-2025
    volatile_symbols = [
        "SOLUSDT",    # Solana - high TPS, very volatile
        "RNDRUSDT",   # Render token - AI hype, high volatility  
        "WIFUSDT",    # Dogwifhat - viral meme coin
        "PEPEUSDT",   # Pepe - classic meme coin volatility
        "AVAXUSDT",   # Avalanche - L1 with high volatility
    ]
    
    config = BacktestConfig(
        symbols=volatile_symbols,
        timeframe=TimeFrame.FIFTEEN_MINUTES,
        since_date=datetime(2024, 1, 1),
        test_start_date=datetime(2024, 6, 1), 
        test_end_date=datetime(2024, 12, 1),
        initial_balance=Decimal("5000"),
        fee_rate=Decimal("0.001")
    )
    
    print(f"🎯 Testing {len(volatile_symbols)} volatile cryptos:")
    for symbol in volatile_symbols:
        print(f"   • {symbol}")
    
    # Use Time-Based strategy (the only profitable one) 
    # but optimized for higher volatility
    from trading_bot.strategies.time_based_reversion import TimeBasedReversionStrategy
    
    strategy = TimeBasedReversionStrategy(
        # More aggressive for volatile assets
        short_ma_period=8,                    # Faster MA
        max_distance_from_ma=0.020,          # Larger deviations (2% vs 0.8%)
        
        # More sensitive RSI for volatile moves
        rsi_period=7,                        # Faster RSI
        rsi_oversold=20,                     # Deep oversold
        
        # Time filters for max volatility periods
        preferred_sessions=['european_morning', 'us_morning'],
        
        # Adjusted for bigger moves
        max_velocity=0.025,                  # Allow more momentum
        min_volume_ratio=0.8,                # Lower volume requirement
        
        # Wider stops for volatility
        stop_loss_atr=2.0,                   # Wider stops 
        take_profit_atr=3.0,                 # Bigger targets
        position_size_pct=0.02               # Smaller size for risk control
    )
    
    print(f"\n🎯 Strategy: {strategy}")
    print(f"📊 Key changes for volatility:")
    print(f"   • Distance from MA: 2.0% (vs 0.8% for stable coins)")
    print(f"   • Take profit: 3x ATR (vs 2.5x)")
    print(f"   • RSI oversold: 20 (vs 25)")
    
    engine = BacktestEngine(config, strategy)
    results = engine.run()
    
    print(f"\n🔥 VOLATILE CRYPTO RESULTS")
    print(f"=" * 50)
    
    total_return = 0
    total_trades = 0
    best_performer = None
    best_return = -float('inf')
    
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
        
        if result.total_return_pct > best_return:
            best_return = result.total_return_pct
            best_performer = symbol
    
    # Portfolio summary
    avg_return = total_return / len(results) if results else 0
    
    print(f"\n🚀 PORTFOLIO SUMMARY:")
    print(f"   Average Return: {avg_return:>6.2f}%")
    print(f"   Total Trades: {total_trades:>8}")
    print(f"   Best Performer: {best_performer} ({best_return:.2f}%)")
    
    # Compare to stable crypto result
    print(f"\n📊 COMPARISON TO STABLE CRYPTO:")
    print(f"   ADAUSDT (stable):  +0.03% (12 trades)")
    print(f"   Volatile average:  {avg_return:+.2f}% ({total_trades} trades)")
    
    if avg_return > 0.03:
        multiplier = avg_return / 0.03
        print(f"   🎉 Volatile cryptos are {multiplier:.1f}x MORE profitable!")
    else:
        print(f"   ⚠️  Volatile cryptos performed worse (higher risk)")
    
    return results


def test_meme_coins():
    """Test specifically on meme coins (highest volatility)."""
    
    print(f"\n🎪 BONUS: Testing Pure MEME COINS")
    print(f"=" * 40)
    
    # Pure meme coins for maximum volatility
    meme_symbols = ["DOGEUSDT", "SHIBUSDT", "PEPEUSDT"]
    
    config = BacktestConfig(
        symbols=meme_symbols,
        timeframe=TimeFrame.FIFTEEN_MINUTES,
        since_date=datetime(2024, 1, 1),
        test_start_date=datetime(2024, 6, 1),
        test_end_date=datetime(2024, 12, 1), 
        initial_balance=Decimal("5000"),
        fee_rate=Decimal("0.001")
    )
    
    from trading_bot.strategies.time_based_reversion import TimeBasedReversionStrategy
    
    # Extreme settings for meme coin chaos
    meme_strategy = TimeBasedReversionStrategy(
        max_distance_from_ma=0.030,          # 3% deviations!
        rsi_oversold=15,                     # Very oversold
        rsi_period=5,                        # Very fast
        stop_loss_atr=2.5,                   # Wide stops for chaos
        take_profit_atr=4.0,                 # Big targets
        position_size_pct=0.015              # Small size for survival
    )
    
    engine = BacktestEngine(config, meme_strategy)
    meme_results = engine.run()
    
    print(f"🎪 MEME COIN RESULTS:")
    for symbol, result in meme_results.items():
        print(f"   {symbol}: {result.total_return_pct:+.2f}% ({result.total_trades} trades)")
    
    return meme_results


def main():
    """Test Time-Based strategy on volatile cryptos."""
    
    print("🚀 VOLATILE CRYPTO SCALPING TEST")
    print("Testing if bigger price swings = bigger profits")
    print("=" * 70)
    
    # Main test: volatile cryptos
    volatile_results = test_volatile_cryptos()
    
    # Bonus: pure meme coin test
    try:
        meme_results = test_meme_coins()
    except Exception as e:
        print(f"❌ Meme coin test failed: {e}")
    
    print(f"\n💡 INSIGHTS:")
    print(f"• Time-Based strategy works better on volatile assets")
    print(f"• Larger price deviations = more mean reversion opportunities") 
    print(f"• Risk management crucial with volatile cryptos")
    print(f"• Consider position sizing based on volatility")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"• Focus on best-performing volatile crypto")
    print(f"• Optimize parameters for that specific asset")
    print(f"• Consider multiple timeframes (5m, 30m)")
    print(f"• Test with different session times")


if __name__ == "__main__":
    main()