import asyncio
from datetime import datetime, timedelta

from trading_bot.core.enums import TimeFrame
from trading_bot.data.market_data import MarketDataManager
from trading_bot.strategies.adaptive_multi_ai_assisted.ai_enhanced_strategy import AIEnhancedStrategy
from trading_bot.strategies.adaptive_multi_ai_assisted.validation_agent import TradeValidationAgent
from trading_bot.strategies.scalping_adaptive import ScalpingAdaptiveStrategy


async def test_structured_ai_validation():
    """Test the new structured AI validation system"""
    
    print("🤖 Testing AI Trade Validation with Structured Actions")
    print("🔬 Real data + Real strategy + Structured AI validation")
    print("=" * 60)
    
    # Initialize data manager
    print("📊 Loading market data...")
    data_manager = MarketDataManager()
    
    symbol = "SOLUSDT"
    timeframe = TimeFrame.FIVE_MINUTES
    
    # Get recent data
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=7 * 24)
    
    dm = MarketDataManager()
    count = dm.download_and_store(symbol, timeframe, 7)
    
    df = data_manager.get_data_for_backtest(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        with_indicators=True
    )
    
    if df.empty:
        print("❌ No data available. Download data first:")
        print(f"   python scripts/download_data.py --symbol {symbol} --timeframe 5m")
        return
    
    print(f"✅ Loaded {len(df):,} candles from {df.index[0]} to {df.index[-1]}")
    
    # Create scalping strategy
    print("⚙️  Initializing scalping strategy...")
    base_strategy = ScalpingAdaptiveStrategy(
        ema_fast=3,
        ema_slow=8,
        rsi_period=7,
        mr_rsi_oversold=50,
        mr_bb_lower=0.4,
        trend_volume_min=0.5,
        base_position_size=0.005
    )
    
    # Prepare data with indicators
    df = base_strategy.prepare_data(df)
    print(f"🔢 Calculated technical indicators")
    
    # Create structured AI-enhanced strategy
    print("🧠 Initializing structured AI validation layer...")
    ai_strategy = AIEnhancedStrategy(
        base_strategy=base_strategy,
        ai_confidence_threshold=6
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
            # Test base strategy signal
            base_signal = base_strategy.buy_condition(row, prev_row)
            
            if base_signal[0]:  # If base strategy finds a signal
                print(f"\n📍 SIGNAL FOUND at {row.name}")
                print(f"   💰 Price: ${row['close']:,.2f}")
                print(f"   🤖 Algorithm: {base_signal[1]}")
                
                # Get structured AI validation
                try:
                    ai_signal = await ai_strategy.buy_condition_with_ai(row, prev_row, df, i)
                    
                    print(f"   🧠 AI Decision: {ai_signal[1] if ai_signal[0] else ai_signal[1]}")
                    
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
    
    # Show detailed AI stats
    ai_strategy.print_ai_stats()
    
    # Test quick structured validation
    print(f"\n🚀 Quick Structured AI Validation Test")
    print("=" * 30)
    try:
        agent = TradeValidationAgent()
        result = await agent.quick_validate_with_actions(
            symbol="SOLUSDT",
            signal_reason="[RANGING] Mean Reversion Oversold",
            current_price=149.65,
            rsi=33.2,
            volume_ratio=0.9,
            regime="ranging"
        )
        
        if result['processing_successful']:
            ai_rec = result['ai_recommendation']
            modified_params = result['modified_params']
            
            print(f"🧠 Structured AI Result:")
            print(f"   Decision: {ai_rec.decision}")
            print(f"   Confidence: {ai_rec.confidence}/10")
            print(f"   Risk Level: {ai_rec.risk_level}")
            print(f"   Actions: {len(ai_rec.recommended_actions)}")
            
            print(f"\n📊 Recommended Actions:")
            for i, action in enumerate(ai_rec.recommended_actions, 1):
                print(f"      {i}. {action.action_type}: {getattr(action, 'reasoning', 'No reasoning provided')}")
            
            print(f"\n⚙️  Applied Modifications:")
            for mod in modified_params.get('modifications_applied', []):
                print(f"      • {mod}")
            
            print(f"\n💰 Trade Parameters:")
            print(f"      Original Entry: ${result['original_params']['entry_price']:.2f}")
            print(f"      Modified Entry: ${modified_params['entry_price']:.2f}")
            print(f"      Original Stop: {result['original_params']['stop_loss_pct']:.3f}")
            print(f"      Modified Stop: {modified_params['stop_loss_pct']:.3f}")
            print(f"      Original Size: ${result['original_params']['position_size']:.2f}")
            print(f"      Modified Size: ${modified_params['position_size']:.2f}")
            print(f"      Should Execute: {'✅ YES' if modified_params['should_execute'] else '❌ NO'}")
        else:
            print(f"❌ Structured validation failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
    
    return test_signals


if __name__ == "__main__":
    async def main():
        try:
            signals = await test_structured_ai_validation()
            
            print(f"\n✅ All tests completed!")
            print(f"💡 Next steps:")
            print(f"   1. Review AI action recommendations")
            print(f"   2. Implement trade parameter modifications")
            print(f"   3. Run backtests with AI-modified parameters")
            print(f"   4. Monitor AI action effectiveness")
            
        except KeyboardInterrupt:
            print(f"\n🛑 Tests interrupted")
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(main())