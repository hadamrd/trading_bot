#!/usr/bin/env python3
"""
Test the optimized EMA Crossover strategy on fresh data
Compare against default parameters to validate optimization
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.strategies.ema_crossover import EMACrossoverStrategy
from trading_bot.core.models import BacktestConfig
from trading_bot.core.enums import TimeFrame


def test_optimized_vs_default():
    """Test optimized EMA parameters vs default parameters"""
    
    print("🧪 TESTING OPTIMIZED EMA STRATEGY")
    print("=" * 50)
    
    # OUT-OF-SAMPLE test period (different from optimization period)
    config = BacktestConfig(
        symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT"],  # Added ADAUSDT for extra validation
        timeframe=TimeFrame.FIFTEEN_MINUTES,
        since_date=datetime(2024, 1, 1),
        test_start_date=datetime(2024, 12, 1),     # Recent period not used in optimization
        test_end_date=datetime(2024, 12, 20),      # Fresh 20 days
        initial_balance=Decimal("10000"),
        fee_rate=Decimal("0.001")
    )
    
    print(f"📊 Test Period: {config.test_start_date.date()} to {config.test_end_date.date()}")
    print(f"🎯 Testing Symbols: {', '.join(config.symbols)}")
    
    # Strategy 1: DEFAULT EMA parameters
    default_strategy = EMACrossoverStrategy(
        fast_period=12,           # Default
        slow_period=26,           # Default
        stop_loss_pct=0.02,       # Default 2%
        take_profit_pct=0.04,     # Default 4%
        position_size_pct=0.1     # Default 10%
    )
    
    # Strategy 2: OPTIMIZED EMA parameters
    optimized_strategy = EMACrossoverStrategy(
        fast_period=20,                    # Optimized: slower
        slow_period=48,                    # Optimized: much slower
        stop_loss_pct=0.04150042940442119, # Optimized: ~4.15%
        take_profit_pct=0.0407410603940765, # Optimized: ~4.07%
        position_size_pct=0.07545506888190803  # Optimized: ~7.55%
    )
    
    print(f"\n🔧 STRATEGY COMPARISON:")
    print(f"                    Default    Optimized")
    print(f"Fast EMA:           12         20")
    print(f"Slow EMA:           26         48") 
    print(f"Stop Loss:          2.0%       4.15%")
    print(f"Take Profit:        4.0%       4.07%")
    print(f"Position Size:      10.0%      7.55%")
    
    # Test both strategies
    strategies = {
        "Default_EMA": default_strategy,
        "Optimized_EMA": optimized_strategy
    }
    
    results = {}
    
    for name, strategy in strategies.items():
        print(f"\n🧪 Testing {name}...")
        
        engine = BacktestEngine(config, strategy)
        strategy_results = engine.run()
        results[name] = strategy_results
        
        # Calculate portfolio metrics
        total_return = sum(r.total_return_pct for r in strategy_results.values())
        total_trades = sum(r.total_trades for r in strategy_results.values())
        avg_win_rate = sum(r.win_rate for r in strategy_results.values()) / len(strategy_results)
        avg_sharpe = sum(r.sharpe_ratio for r in strategy_results.values()) / len(strategy_results)
        max_drawdown = max(r.max_drawdown for r in strategy_results.values())
        
        print(f"📊 {name} Portfolio Results:")
        print(f"   Total Return:    {total_return:>8.2f}%")
        print(f"   Total Trades:    {total_trades:>8}")
        print(f"   Avg Win Rate:    {avg_win_rate:>8.1%}")
        print(f"   Avg Sharpe:      {avg_sharpe:>8.2f}")
        print(f"   Max Drawdown:    {max_drawdown:>8.2f}%")
    
    # Detailed comparison
    print(f"\n📈 DETAILED SYMBOL BREAKDOWN")
    print(f"=" * 60)
    
    for symbol in config.symbols:
        print(f"\n{symbol}:")
        print(f"{'Strategy':<15} {'Return':<8} {'Trades':<7} {'Win%':<6} {'Sharpe':<7}")
        print(f"-" * 50)
        
        for name in strategies.keys():
            result = results[name][symbol]
            print(f"{name:<15} {result.total_return_pct:>6.2f}%  {result.total_trades:>5}   {result.win_rate:>5.1%}  {result.sharpe_ratio:>6.2f}")
    
    # Summary and interpretation
    print(f"\n🎯 OPTIMIZATION VALIDATION")
    print(f"=" * 40)
    
    default_total = sum(r.total_return_pct for r in results["Default_EMA"].values())
    optimized_total = sum(r.total_return_pct for r in results["Optimized_EMA"].values())
    
    improvement = optimized_total - default_total
    improvement_pct = (improvement / abs(default_total)) * 100 if default_total != 0 else 0
    
    print(f"Default EMA Total Return:    {default_total:>6.2f}%")
    print(f"Optimized EMA Total Return:  {optimized_total:>6.2f}%")
    print(f"Improvement:                 {improvement:>+6.2f}% ({improvement_pct:+.1f}%)")
    
    if improvement > 0:
        print(f"✅ OPTIMIZATION SUCCESSFUL! +{improvement:.2f}% improvement")
        print(f"💡 Key insights:")
        print(f"   • Slower EMAs (20/48 vs 12/26) reduced false signals")
        print(f"   • Higher stops (4.15% vs 2%) gave trades more room")
        print(f"   • Balanced risk/reward (~4.15%/4.07%) improved consistency")
        print(f"   • Smaller position size (7.55% vs 10%) reduced risk")
    elif improvement > -1:
        print(f"⚖️  OPTIMIZATION NEUTRAL ({improvement:.2f}% difference)")
        print(f"💡 Both strategies performed similarly - optimization didn't hurt")
    else:
        print(f"❌ OPTIMIZATION FAILED ({improvement:.2f}% worse)")
        print(f"💡 Possible reasons:")
        print(f"   • Overfitting to optimization period")
        print(f"   • Different market conditions in test period") 
        print(f"   • Need more diverse optimization data")
    
    return results


def analyze_parameter_insights():
    """Analyze what the optimization learned"""
    
    print(f"\n🔍 PARAMETER ANALYSIS")
    print(f"=" * 30)
    
    print(f"🎯 What the Optimization Discovered:")
    print()
    print(f"1. SLOWER EMAs (20/48 vs typical 12/26):")
    print(f"   → Reduces noise and false signals")
    print(f"   → Better for crypto's high volatility")
    print(f"   → Captures longer-term trends")
    print()
    print(f"2. HIGHER STOP LOSS (4.15% vs 2%):")
    print(f"   → Gives trades more room to breathe")
    print(f"   → Reduces being stopped out by volatility")
    print(f"   → Better suited for crypto price swings")
    print()
    print(f"3. BALANCED RISK/REWARD (4.15% stop / 4.07% target):")
    print(f"   → Almost 1:1 ratio suggests high win rate strategy")
    print(f"   → Conservative approach to profit taking")
    print(f"   → Prioritizes consistency over big wins")
    print()
    print(f"4. MODERATE POSITION SIZE (7.55% vs 10%):")
    print(f"   → Reduces portfolio risk per trade")
    print(f"   → Allows for more trades over time")
    print(f"   → Better risk management")
    
    print(f"\n📚 STRATEGY CHARACTERISTICS:")
    print(f"   • Type: Conservative trend following")
    print(f"   • Best for: Trending markets with lower noise")
    print(f"   • Risk level: Moderate (smaller positions, wider stops)")
    print(f"   • Expected: Higher win rate, smaller average wins")


def main():
    """Main test function"""
    print("🧪 OPTIMIZED EMA STRATEGY VALIDATION")
    print("Testing on fresh, out-of-sample data")
    print("=" * 60)
    
    try:
        # Test the strategies
        results = test_optimized_vs_default()
        
        # Analyze what we learned
        analyze_parameter_insights()
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. If optimization was successful:")
        print(f"      → Test on even longer periods")
        print(f"      → Try paper trading with these parameters")
        print(f"      → Consider optimizing other strategies")
        print(f"   2. If optimization was neutral/failed:")
        print(f"      → Try different optimization periods")
        print(f"      → Test with more symbols")
        print(f"      → Adjust fitness function weights")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()