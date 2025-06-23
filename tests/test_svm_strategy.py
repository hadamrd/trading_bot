#!/usr/bin/env python3
"""
Test the SVM Sliding Window Strategy
Adaptive ML approach that refits every candle
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from trading_bot.data.market_data import MarketDataManager
from trading_bot.core.enums import TimeFrame

def test_svm_strategy():
    """Test SVM sliding window strategy."""
    print("🤖 SVM SLIDING WINDOW STRATEGY TEST")
    print("=" * 60)
    print("Adaptive ML that refits every candle for current market conditions")
    
    # Load the strategy (need to save it first)
    try:
        
        from trading_bot.strategies.svm_sliding_window import SVMSlidingWindowStrategy
        
        # Test on BTCUSDT with 5-minute timeframe (we have this data)
        symbol = "BTCUSDT"
        timeframe = TimeFrame.FIVE_MINUTES
        
        print(f"🎯 Testing on {symbol} {timeframe.value}")
        
        # Get data
        manager = MarketDataManager()
        df = manager.get_data_for_backtest(
            symbol=symbol,
            timeframe=timeframe,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 11, 1),
            with_indicators=True
        )
        
        if df.empty:
            print(f"❌ No data available for {symbol}")
            return
        
        print(f"📊 Data loaded: {len(df):,} candles")
        print(f"   Period: {df.index[0].date()} to {df.index[-1].date()}")
        
        # Test multiple SVM configurations
        configs = [
            {
                "name": "Conservative",
                "params": {
                    "window_size": 200,
                    "confidence_threshold": 0.70,
                    "retrain_frequency": 1,
                    "position_size_pct": 0.03
                }
            },
            {
                "name": "Balanced",
                "params": {
                    "window_size": 150,
                    "confidence_threshold": 0.65,
                    "retrain_frequency": 1,
                    "position_size_pct": 0.04
                }
            },
            {
                "name": "Aggressive",
                "params": {
                    "window_size": 100,
                    "confidence_threshold": 0.60,
                    "retrain_frequency": 1,
                    "position_size_pct": 0.05
                }
            }
        ]
        
        print(f"\n🧪 Testing {len(configs)} SVM configurations...")
        
        results = {}
        
        for config in configs:
            print(f"\n📊 Testing {config['name']} SVM...")
            
            # Create strategy with config
            strategy = SVMSlidingWindowStrategy(**config['params'])
            
            # Prepare data with strategy indicators
            df_prepared = strategy.prepare_data(df)
            
            # Run custom SVM backtest
            result = strategy.run_svm_backtest(df_prepared)
            results[config['name']] = result
            
            # Print results
            print(f"\n🤖 {config['name']} SVM Results:")
            print(f"   Total Return: {result['total_return_pct']:>8.2f}%")
            print(f"   Total Trades: {result['total_trades']:>8}")
            print(f"   Win Rate: {result['win_rate']:>11.1%}")
            print(f"   Avg Confidence: {result['avg_confidence']:>6.1%}")
            
            if result['total_trades'] > 0:
                avg_per_trade = result['total_return_pct'] / result['total_trades']
                print(f"   Avg per Trade: {avg_per_trade:>7.3f}%")
        
        # Compare to previous strategies
        print(f"\n📊 COMPARISON TO PREVIOUS STRATEGIES")
        print(f"=" * 50)
        print(f"Strategy                Return    Trades   Win Rate")
        print(f"-" * 50)
        print(f"Time-Based (PEPE 15m)    +0.23%      15      86.7%")
        print(f"Time-Based (BTC/ETH 5m)  -0.05%       4      25.0%")
        
        for name, result in results.items():
            win_rate = result['win_rate'] * 100
            print(f"SVM {name:<12}    {result['total_return_pct']:>+6.2f}%    {result['total_trades']:>4}    {win_rate:>6.1f}%")
        
        # Find best performing
        if results:
            best_name = max(results.keys(), key=lambda k: results[k]['total_return_pct'])
            best_result = results[best_name]
            
            print(f"\n🏆 BEST PERFORMER: {best_name} SVM")
            print(f"   Return: {best_result['total_return_pct']:+.2f}%")
            print(f"   Trades: {best_result['total_trades']}")
            
            if best_result['total_return_pct'] > 0.23:
                improvement = (best_result['total_return_pct'] / 0.23) - 1
                print(f"   🎉 {improvement*100:+.0f}% better than previous best!")
        
        manager.close()
        return results
        
    except Exception as e:
        print(f"❌ SVM test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main test function."""
    print("🚀 SVM SLIDING WINDOW STRATEGY")
    print("Adaptive ML that refits every candle")
    print("=" * 60)
    
    results = test_svm_strategy()
    
    if results:
        print(f"\n💡 KEY INSIGHTS:")
        print(f"   • SVM adapts to changing market conditions")
        print(f"   • Higher confidence threshold = fewer but better trades")
        print(f"   • Larger window size = more stable model")
        print(f"   • Feature engineering is crucial for performance")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   • If SVM works well: optimize hyperparameters")
        print(f"   • Try different feature combinations")
        print(f"   • Test on multiple assets")
        print(f"   • Consider ensemble models")
    else:
        print(f"\n❌ SVM test failed - check sklearn installation")
        print(f"Install with: pip install scikit-learn")

if __name__ == "__main__":
    main()