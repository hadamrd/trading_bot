#!/usr/bin/env python3
"""
Debug why the strategy is generating so few signals
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal

import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from trading_bot.data.market_data import MarketDataManager
from trading_bot.core.enums import TimeFrame

def debug_strategy_conditions():
    """Debug why the strategy isn't generating signals"""
    print("🔍 DEBUGGING STRATEGY SIGNAL GENERATION")
    print("=" * 60)
    
    try:
        from trading_bot.strategies.time_based_reversion import TimeBasedReversionStrategy
        
        # Test with current restrictive parameters
        restrictive_strategy = TimeBasedReversionStrategy(
            short_ma_period=8,
            max_distance_from_ma=0.02,  # 2%
            rsi_period=7,
            rsi_oversold=20,            # Very restrictive
            preferred_sessions=['european_morning', 'us_morning'],
            max_velocity=0.025,
            min_volume_ratio=0.8,
            stop_loss_atr=2.0,
            take_profit_atr=3.0,
            position_size_pct=0.02
        )
        
        # Get some real data
        manager = MarketDataManager()
        symbol = "PEPEUSDT"  # One that had 0 trades
        
        df = manager.get_data_for_backtest(
            symbol=symbol,
            timeframe=TimeFrame.FIFTEEN_MINUTES,
            start_date=datetime(2024, 6, 1),
            end_date=datetime(2024, 8, 1),  # Smaller period for debugging
            with_indicators=True
        )
        
        if df.empty:
            print(f"❌ No data for {symbol}")
            return
        
        print(f"📊 Analyzing {len(df)} candles for {symbol}")
        
        # Prepare data with strategy indicators
        df = restrictive_strategy.prepare_data(df)
        
        # Analyze conditions
        analyze_strategy_conditions(df, restrictive_strategy, "RESTRICTIVE")
        
        # Test with more permissive parameters
        print(f"\n" + "="*60)
        print("🎯 TESTING MORE PERMISSIVE PARAMETERS")
        
        permissive_strategy = TimeBasedReversionStrategy(
            short_ma_period=8,
            max_distance_from_ma=0.008,  # 0.8% instead of 2%
            rsi_period=7,
            rsi_oversold=30,             # 30 instead of 20
            preferred_sessions=[],       # Remove time restrictions
            max_velocity=0.05,           # Double the velocity limit
            min_volume_ratio=0.5,        # Lower volume requirement
            stop_loss_atr=2.0,
            take_profit_atr=3.0,
            position_size_pct=0.02
        )
        
        df = permissive_strategy.prepare_data(df)
        analyze_strategy_conditions(df, permissive_strategy, "PERMISSIVE")
        
        manager.close()
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

def analyze_strategy_conditions(df, strategy, label):
    """Analyze how many candles meet each condition"""
    print(f"\n📊 {label} STRATEGY ANALYSIS:")
    print("-" * 40)
    
    total_candles = len(df)
    
    # Check individual conditions
    conditions = {}
    
    # RSI condition
    if 'rsi' in df.columns:
        rsi_condition = df['rsi'] < strategy.rsi_oversold
        conditions['RSI Oversold'] = rsi_condition.sum()
        print(f"   RSI < {strategy.rsi_oversold}: {rsi_condition.sum():,}/{total_candles:,} ({rsi_condition.mean():.1%})")
    
    # Distance from MA condition
    if hasattr(strategy, 'short_ma_period'):
        ma_col = f'sma_{strategy.short_ma_period}'
        if ma_col in df.columns:
            distance = abs(df['close'] - df[ma_col]) / df[ma_col]
            distance_condition = distance > strategy.max_distance_from_ma
            conditions['Distance from MA'] = distance_condition.sum()
            print(f"   Distance > {strategy.max_distance_from_ma:.1%}: {distance_condition.sum():,}/{total_candles:,} ({distance_condition.mean():.1%})")
    
    # Volume condition
    if hasattr(strategy, 'min_volume_ratio') and 'volume_ratio' in df.columns:
        volume_condition = df['volume_ratio'] >= strategy.min_volume_ratio
        conditions['Volume Ratio'] = volume_condition.sum()
        print(f"   Volume >= {strategy.min_volume_ratio}: {volume_condition.sum():,}/{total_candles:,} ({volume_condition.mean():.1%})")
    
    # Velocity condition
    if hasattr(strategy, 'max_velocity'):
        velocity = abs(df['close'].pct_change())
        velocity_condition = velocity <= strategy.max_velocity
        conditions['Max Velocity'] = velocity_condition.sum()
        print(f"   Velocity <= {strategy.max_velocity:.1%}: {velocity_condition.sum():,}/{total_candles:,} ({velocity_condition.mean():.1%})")
    
    # Session condition (if applicable)
    if hasattr(strategy, 'preferred_sessions') and strategy.preferred_sessions:
        # Estimate session condition (simplified)
        df['hour'] = pd.to_datetime(df.index).hour
        session_condition = (
            ((df['hour'] >= 8) & (df['hour'] <= 11)) |   # European morning
            ((df['hour'] >= 14) & (df['hour'] <= 17))    # US morning
        )
        conditions['Preferred Sessions'] = session_condition.sum()
        print(f"   Preferred Sessions: {session_condition.sum():,}/{total_candles:,} ({session_condition.mean():.1%})")
    
    # Combined conditions (simplified estimate)
    print(f"\n💡 ESTIMATED SIGNAL POTENTIAL:")
    if len(conditions) > 0:
        min_signals = min(conditions.values())
        print(f"   Conservative estimate: ~{min_signals} potential signals")
        print(f"   That's {min_signals/total_candles:.2%} of all candles")
    
    # Show actual stats
    print(f"\n📈 DATA STATS:")
    if 'rsi' in df.columns:
        print(f"   RSI range: {df['rsi'].min():.1f} - {df['rsi'].max():.1f}")
        print(f"   RSI median: {df['rsi'].median():.1f}")
    
    if 'volume_ratio' in df.columns:
        print(f"   Volume ratio range: {df['volume_ratio'].min():.2f} - {df['volume_ratio'].max():.2f}")
        print(f"   Volume ratio median: {df['volume_ratio'].median():.2f}")

def suggest_better_parameters():
    """Suggest better parameters based on analysis"""
    print(f"\n🎯 RECOMMENDED PARAMETER ADJUSTMENTS:")
    print("=" * 50)
    
    print(f"For VOLATILE CRYPTOS:")
    print(f"   • RSI oversold: 25-35 (instead of 20)")
    print(f"   • Max distance from MA: 0.5-1.0% (instead of 2.0%)")
    print(f"   • Remove time restrictions for 24/7 crypto markets")
    print(f"   • Min volume ratio: 0.8-1.2 (instead of strict filtering)")
    print(f"   • Max velocity: 0.03-0.08 (crypto moves fast)")
    
    print(f"\nFor MEME COINS (even more volatile):")
    print(f"   • RSI oversold: 20-30")
    print(f"   • Max distance from MA: 1.0-2.0%")
    print(f"   • Higher volume ratios: 1.5+")
    print(f"   • Shorter MA periods: 5-8")

def main():
    """Main debug function"""
    print("🔍 Strategy Signal Generation Debugger")
    
    debug_strategy_conditions()
    suggest_better_parameters()
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Test with more permissive parameters")
    print(f"   2. Focus on one volatile crypto first")
    print(f"   3. Gradually tighten conditions to find sweet spot")
    print(f"   4. Consider different timeframes (5m might be better for scalping)")

if __name__ == "__main__":
    main()