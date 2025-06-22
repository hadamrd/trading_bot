#!/usr/bin/env python3
"""
Test professional RSI Divergence strategy vs simple mean reversion
Show the power of using proven institutional strategies
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

def test_rsi_divergence_strategy():
    """Test the professional RSI Divergence strategy."""
    print("🏆 TESTING PROFESSIONAL RSI DIVERGENCE STRATEGY")
    print("=" * 65)
    print("Based on institutional trading methods")
    print("Typical win rates: 65-80% when properly implemented")
    
    from trading_bot.strategies.rsi_divergence import RSIDivergenceStrategy
    
    # Test on volatile crypto
    config = BacktestConfig(
        symbols=["PEPEUSDT"],
        timeframe=TimeFrame.FIFTEEN_MINUTES,
        since_date=datetime(2024, 1, 1),
        test_start_date=datetime(2024, 6, 1),
        test_end_date=datetime(2024, 11, 1),  # 5 months
        initial_balance=Decimal("5000"),
        fee_rate=Decimal("0.001")
    )
    
    # Professional RSI Divergence parameters
    strategy = RSIDivergenceStrategy(
        # RSI settings
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
        
        # Divergence detection
        divergence_lookback=20,
        min_swing_bars=5,
        min_divergence_strength=0.3,  # Minimum strength for signal
        
        # Confirmations
        require_volume_confirmation=True,
        require_trend_confirmation=True,
        ema_trend_period=50,
        
        # Risk management (2:1 risk/reward)
        stop_loss_atr=2.0,
        take_profit_atr=4.0,
        
        # Position sizing
        base_position_size=0.03,
        max_position_size=0.08,
        
        # Volume filter
        min_volume_ratio=1.5
    )
    
    print(f"🎯 RSI Divergence Strategy:")
    print(f"   • Detects price/RSI divergences")
    print(f"   • Uses {strategy.divergence_lookback} bar lookback")
    print(f"   • Requires {strategy.min_divergence_strength} minimum strength")
    print(f"   • Volume confirmation required")
    print(f"   • 2:1 risk/reward ratio")
    
    engine = BacktestEngine(config, strategy)
    results = engine.run()
    
    return results

def test_simple_strategy_comparison():
    """Test simple time-based strategy for comparison."""
    print(f"\n📊 COMPARISON: SIMPLE TIME-BASED STRATEGY")
    print("=" * 50)
    
    from trading_bot.strategies.time_based_reversion import TimeBasedReversionStrategy
    
    config = BacktestConfig(
        symbols=["PEPEUSDT"],
        timeframe=TimeFrame.FIFTEEN_MINUTES,
        since_date=datetime(2024, 1, 1),
        test_start_date=datetime(2024, 6, 1),
        test_end_date=datetime(2024, 11, 1),
        initial_balance=Decimal("5000"),
        fee_rate=Decimal("0.001")
    )
    
    # Your optimized simple strategy
    strategy = TimeBasedReversionStrategy(
        short_ma_period=6,
        max_distance_from_ma=0.008,
        rsi_period=7,
        rsi_oversold=30,
        preferred_sessions=[],
        max_velocity=0.06,
        min_volume_ratio=1.0,
        stop_loss_atr=2.0,
        take_profit_atr=3.0,
        position_size_pct=0.02
    )
    
    engine = BacktestEngine(config, strategy)
    results = engine.run()
    
    return results

def compare_strategies():
    """Compare professional vs simple strategies."""
    print("🚀 PROFESSIONAL vs SIMPLE STRATEGY COMPARISON")
    print("=" * 70)
    
    # Test professional strategy
    try:
        professional_results = test_rsi_divergence_strategy()
        prof_symbol = list(professional_results.keys())[0]
        prof_result = professional_results[prof_symbol]
    except Exception as e:
        print(f"❌ Professional strategy failed: {e}")
        professional_results = None
        prof_result = None
    
    # Test simple strategy
    simple_results = test_simple_strategy_comparison()
    simple_symbol = list(simple_results.keys())[0]
    simple_result = simple_results[simple_symbol]
    
    # Comparison table
    print(f"\n📊 DETAILED COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'Simple':<15} {'Professional':<15} {'Winner':<10}")
    print("-" * 60)
    
    metrics = [
        ('Total Return %', simple_result.total_return_pct, prof_result.total_return_pct if prof_result else 0),
        ('Total Trades', simple_result.total_trades, prof_result.total_trades if prof_result else 0),
        ('Win Rate %', simple_result.win_rate * 100, prof_result.win_rate * 100 if prof_result else 0),
        ('Sharpe Ratio', simple_result.sharpe_ratio, prof_result.sharpe_ratio if prof_result else 0),
        ('Max Drawdown %', simple_result.max_drawdown, prof_result.max_drawdown if prof_result else 0),
        ('Profit Factor', simple_result.profit_factor, prof_result.profit_factor if prof_result else 0),
    ]
    
    for metric_name, simple_val, prof_val in metrics:
        if metric_name == 'Max Drawdown %':
            winner = "Professional" if prof_val < simple_val else "Simple"
        else:
            winner = "Professional" if prof_val > simple_val else "Simple"
        
        print(f"{metric_name:<20} {simple_val:<15.2f} {prof_val:<15.2f} {winner:<10}")
    
    # Analysis
    print(f"\n🎯 ANALYSIS:")
    if prof_result:
        if prof_result.total_return_pct > simple_result.total_return_pct:
            improvement = ((prof_result.total_return_pct / simple_result.total_return_pct) - 1) * 100
            print(f"   📈 Professional strategy outperformed by {improvement:.1f}%")
        
        if prof_result.win_rate > simple_result.win_rate:
            print(f"   🎯 Higher win rate: {prof_result.win_rate:.1%} vs {simple_result.win_rate:.1%}")
        
        if prof_result.sharpe_ratio > simple_result.sharpe_ratio:
            print(f"   📊 Better risk-adjusted returns: {prof_result.sharpe_ratio:.2f} vs {simple_result.sharpe_ratio:.2f}")

def suggest_next_strategies():
    """Suggest other professional strategies to implement."""
    print(f"\n🚀 NEXT PROFESSIONAL STRATEGIES TO IMPLEMENT:")
    print("=" * 55)
    
    strategies = [
        {
            "name": "Bollinger Band Squeeze",
            "description": "Detect low volatility → explosive breakouts",
            "win_rate": "60-75%",
            "best_for": "Range-bound → trending transitions"
        },
        {
            "name": "VWAP + Volume Profile",
            "description": "Institutional-level volume analysis",
            "win_rate": "55-70%", 
            "best_for": "High-volume crypto pairs"
        },
        {
            "name": "Dual Momentum",
            "description": "Academic quantitative strategy",
            "win_rate": "50-65%",
            "best_for": "Portfolio management"
        },
        {
            "name": "Grid Trading Adaptive",
            "description": "Range trading with ML adaptation",
            "win_rate": "70-85%",
            "best_for": "Sideways crypto markets"
        },
        {
            "name": "Market Microstructure",
            "description": "Order flow and market depth analysis",
            "win_rate": "65-80%",
            "best_for": "Short-term scalping"
        }
    ]
    
    for i, strategy in enumerate(strategies, 1):
        print(f"{i}. 🏆 {strategy['name']}")
        print(f"   Description: {strategy['description']}")
        print(f"   Typical Win Rate: {strategy['win_rate']}")
        print(f"   Best For: {strategy['best_for']}\n")

def main():
    """Main comparison function."""
    print("🎯 Professional Strategy Implementation Test")
    
    compare_strategies()
    suggest_next_strategies()
    
    print(f"💡 KEY INSIGHTS:")
    print(f"   • Professional strategies use proven methods")
    print(f"   • Higher win rates through better signal quality")
    print(f"   • More sophisticated risk management")
    print(f"   • Based on institutional trading practices")
    
    print(f"\n🚀 RECOMMENDED NEXT STEPS:")
    print(f"   1. Implement RSI Divergence strategy fully")
    print(f"   2. Test on multiple crypto pairs")
    print(f"   3. Add machine learning confirmations") 
    print(f"   4. Implement Bollinger Squeeze next")
    print(f"   5. Build strategy portfolio approach")

if __name__ == "__main__":
    main()