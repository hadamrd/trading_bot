#!/usr/bin/env python3
"""
Test AI Trade Validation with Real Data
Shows how to integrate AI validation with existing trading strategy
"""

import sys
import os
import asyncio
from pathlib import Path

from trading_bot.strategies.adaptive_multi_ai_assisted.ai_enhanced_strategy import AIEnhancedStrategy
from trading_bot.strategies.adaptive_multi_ai_assisted.trade_validation_agent import TradeValidationAgent

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd

# Import existing trading components
from trading_bot.data.market_data import MarketDataManager
from trading_bot.core.enums import TimeFrame
from trading_bot.strategies.adaptive_multi import AdaptiveMultiStrategy


async def test_ai_validation_integration():
    """Test AI validation with real market data and existing strategy"""
    
    print("🤖 Testing AI Trade Validation Integration")
    print("🔬 Real data + Real strategy + AI validation")
    print("=" * 60)
    
    # Initialize data manager
    print("📊 Loading market data...")
    data_manager = MarketDataManager()
    
    symbol = "BTCUSDT"
    timeframe = TimeFrame.FIVE_MINUTES
    
    # Get recent data
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=24)  # Last 24 hours
    
    df = data_manager.get_data_for_backtest(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        with_indicators=True
    )
    
    if df.empty:
        print("❌ No data available. Download data first:")
        print("   python scripts/download_data.py --symbol BTCUSDT --timeframe 5m")
        return
    
    print(f"✅ Loaded {len(df):,} candles from {df.index[0]} to {df.index[-1]}")
    
    # Create base strategy
    print("⚙️  Initializing adaptive strategy...")
    base_strategy = AdaptiveMultiStrategy(
        lookback_period=50,
        volatility_lookback=200,
        base_position_size=0.02
    )
    
    # Prepare data with indicators
    df = base_strategy.prepare_data(df)
    print(f"🔢 Calculated technical indicators")
    
    # Create AI-enhanced strategy
    print("🧠 Initializing AI validation layer...")
    ai_strategy = AIEnhancedStrategy(
        base_strategy=base_strategy,
        use_ai_validation=True,
        ai_threshold_confidence=6
    )
    
    # Test on recent candles
    print(f"\n🔍 Testing on recent market data...")
    test_signals = []
    
    # Look for signals in recent data
    for i in range(len(df) - 50, len(df) - 1):
        if i <= 0:
            continue
            
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        try:
            # Test both base strategy and AI-enhanced strategy
            base_signal = base_strategy.buy_condition(row, prev_row)
            
            if base_signal[0]:  # If base strategy finds a signal
                print(f"\n📍 SIGNAL FOUND at {row.name}")
                print(f"   💰 Price: ${row['close']:,.2f}")
                print(f"   🤖 Algorithm: {base_signal[1]}")
                
                # Get AI validation
                try:
                    ai_signal = await ai_strategy.buy_condition(row, prev_row, df, i)
                    
                    print(f"   🧠 AI Decision: {ai_signal[1] if ai_signal[0] else 'REJECTED'}")
                    
                    # Store for analysis
                    test_signals.append({
                        'timestamp': row.name,
                        'price': row['close'],
                        'algorithm_signal': base_signal[1],
                        'ai_approved': ai_signal[0],
                        'ai_reason': ai_signal[1],
                        'rsi': row.get('rsi', 0),
                        'volume_ratio': row.get('volume_ratio', 1.0),
                        'regime': getattr(base_strategy, 'current_regime', 'unknown')
                    })
                    
                except Exception as e:
                    print(f"   ❌ AI Validation Error: {e}")
                    
        except Exception as e:
            print(f"   ⚠️  Error processing candle {i}: {e}")
    
    # Analysis
    print(f"\n📈 SIGNAL ANALYSIS")
    print("=" * 40)
    
    if test_signals:
        total_algorithm_signals = len(test_signals)
        ai_approved = len([s for s in test_signals if s['ai_approved']])
        ai_rejected = total_algorithm_signals - ai_approved
        
        print(f"🤖 Total Algorithm Signals: {total_algorithm_signals}")
        print(f"✅ AI Approved: {ai_approved} ({ai_approved/total_algorithm_signals*100:.1f}%)")
        print(f"❌ AI Rejected: {ai_rejected} ({ai_rejected/total_algorithm_signals*100:.1f}%)")
        
        print(f"\n📋 SIGNAL DETAILS:")
        for i, signal in enumerate(test_signals[-3:], 1):  # Show last 3
            status = "✅ APPROVED" if signal['ai_approved'] else "❌ REJECTED"
            print(f"   {i}. {status}")
            print(f"      Time: {signal['timestamp'].strftime('%H:%M:%S')}")
            print(f"      Algorithm: {signal['algorithm_signal']}")
            print(f"      AI: {signal['ai_reason']}")
            print(f"      Context: RSI={signal['rsi']:.1f}, Vol={signal['volume_ratio']:.1f}x")
    
    else:
        print("⚠️  No signals found in recent data")
        print("💡 This could mean:")
        print("   • Market conditions didn't trigger signals")
        print("   • Strategy parameters are conservative")
        print("   • Try testing on more volatile periods")
    
    # Show AI stats
    ai_strategy.print_ai_stats()
    
    return test_signals


async def quick_ai_test():
    """Quick test of AI validation without full data"""
    
    print("\n🚀 Quick AI Validation Test")
    print("=" * 30)
    
    
    try:
        # Initialize AI validator
        ai_validator = TradeValidationAgent()
        
        # Test quick validation
        response = await ai_validator.quick_validate(
            symbol="BTCUSDT",
            signal_reason="[RANGING] Mean Reversion Oversold",
            current_price=107250.0,
            rsi=28.5,
            volume_ratio=1.8,
            regime="ranging"
        )
        
        print(f"🧠 AI Validation Result:")
        print(f"   Decision: {response.decision.value}")
        print(f"   Confidence: {response.confidence}/10")
        print(f"   Reasoning: {response.primary_reasoning}")
        print(f"   Risk Level: {response.risk_level.value}")
        print(f"   Technical Quality: {response.technical_quality.value}")
        
        if response.risk_factors:
            print(f"   Risk Factors: {', '.join(response.risk_factors)}")
        
        if response.suggested_modifications:
            print(f"   Suggestions: {response.suggested_modifications}")
        
        print(f"✅ AI quick test completed!")
        
    except Exception as e:
        print(f"❌ AI test failed: {e}")
        print(f"💡 Make sure you have Claude API key configured")


async def compare_with_without_ai():
    """Compare strategy performance with and without AI validation"""
    
    print(f"\n📊 COMPARISON: With vs Without AI")
    print("=" * 40)
    print(f"This would show:")
    print(f"   📈 Win rate improvement")
    print(f"   💰 Return enhancement") 
    print(f"   📉 Drawdown reduction")
    print(f"   🎯 Signal quality improvement")
    print(f"")
    print(f"💡 To implement:")
    print(f"   1. Run backtest with base strategy")
    print(f"   2. Run backtest with AI-enhanced strategy")
    print(f"   3. Compare results")


if __name__ == "__main__":
    async def main():
        try:
            # Test AI integration with real data
            signals = await test_ai_validation_integration()
            
            # Quick AI test
            await quick_ai_test()
            
            # Show comparison concept
            await compare_with_without_ai()
            
            print(f"\n✅ All tests completed!")
            print(f"💡 Next steps:")
            print(f"   1. Configure Claude API key in your settings")
            print(f"   2. Run full backtest comparison")
            print(f"   3. Tune AI confidence thresholds")
            print(f"   4. Monitor AI validation performance")
            
        except KeyboardInterrupt:
            print(f"\n🛑 Tests interrupted")
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(main())
