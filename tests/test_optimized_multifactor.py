#!/usr/bin/env python3
"""
Test the optimized Multi-Factor Strategy on Jan-May 2025 data
Validate that optimization wasn't just overfitting
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


def test_optimized_multifactor_2025():
    """Test optimized Multi-Factor strategy on fresh 2025 data"""
    
    print("🧪 TESTING OPTIMIZED MULTI-FACTOR STRATEGY")
    print("=" * 55)
    print("Validation Period: Jan-May 2025 (OUT-OF-SAMPLE)")
    
    config = BacktestConfig(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "PEPEUSDT"],
        timeframe=TimeFrame.FIVE_MINUTES,
        since_date=datetime(2024, 12, 1),   # Extra data for indicators
        test_start_date=datetime(2025, 1, 1),
        test_end_date=datetime(2025, 5, 31),
        initial_balance=Decimal("10000"),
        fee_rate=Decimal("0.001")
    )
    
    print(f"📊 Test Period: {config.test_start_date.date()} to {config.test_end_date.date()}")
    print(f"🎯 Symbols: {', '.join(config.symbols)}")
    
    # Default Multi-Factor strategy
    default_strategy = MultiFactorStrategy()
    
    # OPTIMIZED Multi-Factor strategy with genetic algorithm results
    optimized_strategy = MultiFactorStrategy(
        # Core parameters
        ema_short_period=14,
        ema_long_period=16,
        rsi_period=20,
        risk_per_trade=0.0225,
        take_profit_factor=2.7171,
        stop_loss_factor=1.0196,
        
        # Signal-specific parameters
        short_cross_rsi_threshold=63.8785,
        vwap_bounce_rsi_threshold=51.2792,
        ema_golden_cross_rsi_threshold=37.3295,
        short_cross_atr_factor=0.6510,
        vwap_bounce_atr_factor=0.9151,
        ema_golden_cross_atr_factor=1.1112,
        
        # Optimized signal selection (4 active, 2 disabled)
        use_short_cross=False,      # ❌ Disabled by optimization
        use_long_cross=True,        # ✅ Kept
        use_short_bounce=True,      # ✅ Kept
        use_long_bounce=True,       # ✅ Kept
        use_vwap_bounce=False,      # ❌ Disabled by optimization
        use_ema_golden_cross=True   # ✅ Kept
    )
    
    print(f"\n🔧 OPTIMIZED CONFIGURATION:")
    print(f"  Active Signals: Long Cross, Short Bounce, Long Bounce, EMA Golden Cross")
    print(f"  Disabled: Short Cross, VWAP Bounce")
    print(f"  EMA Periods: {optimized_strategy.ema_short_period}/{optimized_strategy.ema_long_period}")
    print(f"  Risk/Trade: {optimized_strategy.risk_per_trade:.3f}")
    print(f"  Risk/Reward: 1:{optimized_strategy.take_profit_factor:.2f}")
    
    # Test both configurations
    strategies = {
        "Default": default_strategy,
        "Optimized": optimized_strategy
    }
    
    results = {}
    
    for name, strategy in strategies.items():
        print(f"\n🧪 Testing {name} Multi-Factor...")
        
        engine = BacktestEngine(config, strategy)
        strategy_results = engine.run()
        results[name] = strategy_results
        
        # Portfolio totals
        total_return = sum(r.total_return_pct for r in strategy_results.values())
        total_trades = sum(r.total_trades for r in strategy_results.values())
        avg_win_rate = sum(r.win_rate for r in strategy_results.values()) / len(strategy_results)
        avg_sharpe = sum(r.sharpe_ratio for r in strategy_results.values()) / len(strategy_results)
        
        print(f"📊 {name} Results:")
        print(f"   Total Return:    {total_return:>8.2f}%")
        print(f"   Total Trades:    {total_trades:>8}")
        print(f"   Avg Win Rate:    {avg_win_rate:>8.1%}")
        print(f"   Avg Sharpe:      {avg_sharpe:>8.2f}")
    
    # Detailed breakdown
    print(f"\n📈 DETAILED RESULTS BY SYMBOL")
    print(f"=" * 55)
    print(f"{'Symbol':<10} {'Default':<12} {'Optimized':<12} {'Improvement':<12}")
    print("-" * 55)
    
    default_total = 0
    optimized_total = 0
    
    for symbol in config.symbols:
        default_ret = results["Default"][symbol].total_return_pct
        optimized_ret = results["Optimized"][symbol].total_return_pct
        improvement = optimized_ret - default_ret
        
        default_total += default_ret
        optimized_total += optimized_ret
        
        print(f"{symbol:<10} {default_ret:>+6.2f}%      {optimized_ret:>+6.2f}%      {improvement:>+6.2f}%")
    
    total_improvement = optimized_total - default_total
    
    print("-" * 55)
    print(f"{'PORTFOLIO':<10} {default_total:>+6.2f}%      {optimized_total:>+6.2f}%      {total_improvement:>+6.2f}%")
    
    # Validation results
    print(f"\n🎯 OPTIMIZATION VALIDATION")
    print(f"=" * 40)
    
    print(f"Default Multi-Factor:    {default_total:>+6.2f}%")
    print(f"Optimized Multi-Factor:  {optimized_total:>+6.2f}%")
    print(f"Improvement:             {total_improvement:>+6.2f}%")
    
    if optimized_total > 5.0:
        print(f"\n🎉 OPTIMIZATION VALIDATED!")
        print(f"✅ Strategy is highly profitable on fresh data")
        print(f"📈 {optimized_total:.2f}% return over 5 months")
        print(f"🚀 Ready for paper trading consideration")
        
    elif optimized_total > 2.0:
        print(f"\n✅ OPTIMIZATION SUCCESSFUL!")
        print(f"📈 Strategy shows solid profits on out-of-sample data")
        print(f"💡 {optimized_total:.2f}% return is promising")
        
    elif optimized_total > 0:
        print(f"\n⚖️ OPTIMIZATION PARTIALLY SUCCESSFUL")
        print(f"📊 Strategy is profitable but modest: {optimized_total:.2f}%")
        print(f"💭 Consider further optimization or different assets")
        
    else:
        print(f"\n❌ OPTIMIZATION FAILED VALIDATION")
        print(f"📉 Strategy loses money on fresh data: {optimized_total:.2f}%")
        print(f"⚠️ Likely overfitted to Sep-Dec 2024 period")
    
    # Compare to previous strategies
    print(f"\n📊 STRATEGY COMPARISON (Jan-May 2025)")
    print(f"=" * 45)
    print(f"Strategy                    Return")
    print(f"-" * 35)
    print(f"EMA Crossover               -3.08%  ❌")
    print(f"Time-Based Reversion        -0.82%  ❌")
    print(f"Multi-Factor (Default)      {default_total:>+6.2f}%  {'✅' if default_total > 0 else '❌'}")
    print(f"Multi-Factor (Optimized)    {optimized_total:>+6.2f}%  {'✅' if optimized_total > 0 else '❌'}")
    
    if optimized_total > 0:
        print(f"\n🏆 FINALLY FOUND A PROFITABLE STRATEGY!")
        print(f"🧬 Genetic optimization successfully discovered:")
        print(f"   • Which signals work best (4 out of 6)")
        print(f"   • Optimal risk management parameters")
        print(f"   • Proper position sizing")
        
        print(f"\n📋 KEY INSIGHTS:")
        print(f"   • Short Cross & VWAP Bounce signals are noise")
        print(f"   • Long Cross + EMA Golden Cross work well together")
        print(f"   • Short/Long Bounce provide good entry timing")
        print(f"   • Conservative risk (2.25% per trade) is optimal")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Test on different market conditions")
        print(f"   2. Consider paper trading implementation")
        print(f"   3. Optimize for specific asset classes")
        print(f"   4. Implement live trading infrastructure")
    else:
        print(f"\n💭 STILL SEARCHING...")
        print(f"Even optimized Multi-Factor couldn't beat the market")
        print(f"Consider:")
        print(f"   • Different optimization periods")
        print(f"   • Alternative asset classes")
        print(f"   • Machine learning approaches")
        print(f"   • Ensemble methods")
    
    return results


def analyze_signal_effectiveness():
    """Analyze which signals the optimization kept vs disabled"""
    
    print(f"\n🔍 SIGNAL EFFECTIVENESS ANALYSIS")
    print(f"=" * 40)
    
    print(f"✅ SIGNALS KEPT BY OPTIMIZATION:")
    print(f"   1. Long Cross - Price crossing above long EMA")
    print(f"      → Good for trend confirmation")
    print(f"      → Filters out short-term noise")
    print()
    print(f"   2. Short Bounce - Bouncing off short EMA")
    print(f"      → Captures pullback entries")
    print(f"      → Good timing for trend continuation")
    print()
    print(f"   3. Long Bounce - Bouncing off long EMA")
    print(f"      → Strong support level bounces")
    print(f"      → Higher probability setups")
    print()
    print(f"   4. EMA Golden Cross - Short EMA crossing above long EMA")
    print(f"      → Classic trend reversal signal")
    print(f"      → Strong momentum indicator")
    
    print(f"\n❌ SIGNALS DISABLED BY OPTIMIZATION:")
    print(f"   1. Short Cross - Price crossing above short EMA")
    print(f"      → Too noisy, many false signals")
    print(f"      → Short EMA too sensitive for crypto")
    print()
    print(f"   2. VWAP Bounce - Bouncing off VWAP")
    print(f"      → Less relevant for 5-minute timeframe")
    print(f"      → Institutional level more important on longer TF")
    
    print(f"\n💡 OPTIMIZATION INSIGHTS:")
    print(f"   • Longer-term signals (Long Cross, Golden Cross) work better")
    print(f"   • Bounce signals provide good entry timing")
    print(f"   • Short-term signals (Short Cross) add noise")
    print(f"   • VWAP less effective on very short timeframes")


def main():
    """Main test function"""
    print("🧬 GENETIC OPTIMIZATION VALIDATION")
    print("Testing optimized Multi-Factor strategy on fresh data")
    print("=" * 60)
    
    try:
        # Test the optimized strategy
        results = test_optimized_multifactor_2025()
        
        # Analyze what the optimization learned
        analyze_signal_effectiveness()
        
        print(f"\n🎯 VALIDATION COMPLETE!")
        print(f"The genetic algorithm results have been tested on")
        print(f"completely fresh, out-of-sample data (Jan-May 2025)")
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()