#!/usr/bin/env python3
"""
Quick test of the Adaptive Multi-Strategy on 5-minute data
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import datetime, timedelta
from decimal import Decimal

from trading_bot.data.market_data import MarketDataManager
from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.core.models import BacktestConfig
from trading_bot.core.enums import TimeFrame
from trading_bot.strategies.adaptive_multi import AdaptiveMultiStrategy


def test_adaptive_strategy():
    """Test the adaptive strategy on recent 5-minute data"""
    
    print("🤖 Testing Adaptive Multi-Strategy System")
    print("⏰ Timeframe: 5-minute candles")
    print("🎯 Focus: Regime switching with volatility-based sizing")
    print("=" * 60)
    
    # Initialize data manager
    data_manager = MarketDataManager()
    
    # Test on BTCUSDT with 5-minute data
    symbol = "BTCUSDT"
    timeframe = TimeFrame.FIVE_MINUTES
    
    print(f"📥 Checking {symbol} {timeframe.value} data availability...")
    
    # Check if we have recent data
    info = data_manager.get_data_info(symbol, timeframe)
    print(f"📊 Current data: {info['candle_count']:,} candles")
    
    if info['candle_count'] < 1000:  # Need at least ~3 days of 5min data
        print(f"📥 Downloading recent data...")
        data_manager.download_and_store(symbol, timeframe, days=7)
    
    # Get test period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)  # 3 days of 5min data
    
    print(f"🕒 Test period: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
    
    # Create backtest config
    config = BacktestConfig(
        symbols=[symbol],
        timeframe=timeframe,
        since_date=start_date - timedelta(hours=8),  # Extra for indicators
        test_start_date=start_date,
        test_end_date=end_date,
        initial_balance=Decimal("5000"),
        fee_rate=Decimal("0.001")
    )
    
    # Create adaptive strategy
    strategy = AdaptiveMultiStrategy(
        lookback_period=50,          # ~4 hours of 5min candles
        volatility_lookback=200,     # ~16 hours for volatility context
        base_position_size=0.02,     # 2% base position size
        
        # Tune for 5-minute trading
        trending_adx_threshold=25,
        breakout_volatility_threshold=0.75,
        mr_rsi_oversold=30,
        mr_rsi_overbought=75
    )
    
    print(f"⚙️  Strategy configured:")
    print(f"   📊 Lookback: {strategy.lookback_period} candles (~{strategy.lookback_period*5/60:.1f} hours)")
    print(f"   💰 Base position: {strategy.base_position_size*100:.1f}% of balance")
    print(f"   🎯 3 strategies: Trend/Range/Breakout")
    print(f"   😌 4 moods: Calm/Normal/Nervous/Panic")
    
    # Run backtest
    print(f"\n🧪 Running backtest...")
    engine = BacktestEngine(config, strategy)
    results = engine.run()
    
    # Analyze results
    result = results[symbol]
    
    print(f"\n📈 ADAPTIVE STRATEGY RESULTS")
    print("=" * 50)
    print(f"📊 Total Trades: {result.total_trades}")
    print(f"🎯 Win Rate: {result.win_rate:.1%}")
    print(f"💰 Total Return: ${result.total_return:.2f} ({result.total_return_pct:.2f}%)")
    print(f"📉 Max Drawdown: {result.max_drawdown:.2f}%")
    print(f"⚖️  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    
    if result.total_trades > 0:
        print(f"🏭 Profit Factor: {result.profit_factor:.2f}")
        print(f"⏱️  Avg Hold Time: {result.average_holding_time:.1f} hours")
        
        # Analyze strategy distribution
        strategy_usage = {}
        mood_usage = {}
        
        for trade in result.trades:
            reason = trade.buy_reason
            
            # Extract regime
            if "[" in reason and "]" in reason:
                regime = reason.split("]")[0].split("[")[1]
                strategy_usage[regime] = strategy_usage.get(regime, 0) + 1
        
        if strategy_usage:
            print(f"\n🎯 STRATEGY USAGE:")
            total_trades = sum(strategy_usage.values())
            for regime, count in sorted(strategy_usage.items()):
                pct = (count / total_trades) * 100
                print(f"   {regime}: {count} trades ({pct:.1f}%)")
        
        # Show performance by regime
        if len(strategy_usage) > 1:
            print(f"\n📊 REGIME PERFORMANCE:")
            
            regime_stats = {}
            for trade in result.trades:
                reason = trade.buy_reason
                if "[" in reason and "]" in reason:
                    regime = reason.split("]")[0].split("[")[1]
                    
                    if regime not in regime_stats:
                        regime_stats[regime] = {'wins': 0, 'total': 0, 'profit': 0}
                    
                    regime_stats[regime]['total'] += 1
                    regime_stats[regime]['profit'] += trade.profit
                    if trade.profit > 0:
                        regime_stats[regime]['wins'] += 1
            
            for regime, stats in regime_stats.items():
                win_rate = (stats['wins'] / stats['total']) * 100 if stats['total'] > 0 else 0
                print(f"   {regime}: {win_rate:.1f}% win rate, ${stats['profit']:.2f} P&L")
        
        # Show recent trades
        print(f"\n🔄 RECENT TRADES (last 5):")
        for trade in result.trades[-5:]:
            profit_pct = trade.return_percentage
            status = "✅" if profit_pct > 0 else "❌"
            duration_min = trade.duration_hours * 60
            
            print(f"   {status} {trade.buy_reason}")
            print(f"      ${trade.open_price:.2f} → ${trade.close_price:.2f} "
                  f"({profit_pct:+.2f}%) | {duration_min:.0f}min")
    
    else:
        print(f"\n⚠️  No trades executed during test period")
        print(f"💡 This might indicate:")
        print(f"   • Market conditions didn't trigger strategy signals")
        print(f"   • Strategy parameters too conservative")  
        print(f"   • Need longer test period or different market conditions")
    
    print(f"\n💡 VOLATILITY MOOD SYSTEM:")
    print(f"   😎 CALM (low vol): 140% position size")
    print(f"   😐 NORMAL: 100% position size") 
    print(f"   😰 NERVOUS (high vol): 60% position size")
    print(f"   😱 PANIC (extreme vol): 30% position size")
    print(f"   → Maintains consistent risk across market conditions")
    
    return result


def quick_regime_explanation():
    """Explain the regime detection logic"""
    
    print(f"\n🧠 REGIME DETECTION LOGIC:")
    print("=" * 40)
    print(f"🔄 TRENDING:")
    print(f"   • ADX > 25 (strong directional movement)")
    print(f"   • EMA trend strength > 0.8%")
    print(f"   • Strategy: EMA crossover with volume confirmation")
    print()
    print(f"📊 RANGING:")
    print(f"   • Lower ADX or weak trends")
    print(f"   • Default mode for choppy markets")
    print(f"   • Strategy: Bollinger Band + RSI mean reversion")
    print()
    print(f"🚀 BREAKOUT:")
    print(f"   • Volatility > 80th percentile")
    print(f"   • Price momentum > 1.5%")
    print(f"   • Strategy: High volume momentum breakouts")
    print()
    print(f"🎯 The system automatically switches between these")
    print(f"   strategies based on real-time market conditions!")


if __name__ == "__main__":
    try:
        # Run the test
        result = test_adaptive_strategy()
        
        # Show explanation
        quick_regime_explanation()
        
        print(f"\n✅ Test completed successfully!")
        print(f"💡 Next steps:")
        print(f"   1. Try different symbols (ETHUSDT, ADAUSDT, etc.)")
        print(f"   2. Adjust regime thresholds for your market")
        print(f"   3. Optimize parameters using genetic algorithm") 
        print(f"   4. Test on longer periods for more trades")
        
    except Exception as e:
        print(f"\n❌ Error running test: {e}")
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Make sure ClickHouse is running: docker ps")
        print(f"   2. Check data availability: python scripts/show_data_status.py")
        print(f"   3. Download data: python scripts/download_data.py --symbol BTCUSDT --timeframe 5m")
        
        import traceback
        traceback.print_exc()