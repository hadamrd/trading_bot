#!/usr/bin/env python3
"""
Test Runner - Quick test of the complete trading system
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from trading_bot.order_book_trading.strats.simple_strat import SimpleOrderBookStrategy
from trading_bot.order_book_trading.trading_engine import LiveTradingEngine


async def quick_test():
    """Run a quick 2-minute test of the system"""
    
    print("🧪 Quick Trading System Test")
    print("=" * 40)
    print("This will run for 2 minutes to test the system")
    print("You should see:")
    print("  📡 Connection to Binance")
    print("  📊 Market data updates") 
    print("  🎯 Trading signals (maybe)")
    print("  📈 Periodic stats")
    print("\nPress Ctrl+C anytime to stop early\n")
    
    # Create a conservative strategy for testing
    strategy = SimpleOrderBookStrategy(
        position_size=50,      # $50 trades
        min_wall_size=200000,  # Much higher wall threshold (200k)
        min_pressure_diff=0.20, # Strong pressure needed (20% deviation)
        cooldown_seconds=120,   # 2 minute cooldown between trades
        min_hold_seconds=60     # Hold positions for at least 1 minute
    )
    
    # Create engine with frequent stats
    engine = LiveTradingEngine(
        strategy=strategy,
        initial_balance=200,   # Small test balance
        stats_interval=15      # Stats every 15 seconds
    )
    
    try:
        # Run for limited time or until Ctrl+C
        await asyncio.wait_for(
            engine.start("WIFUSDT"), 
            timeout=120  # 2 minutes
        )
    except asyncio.TimeoutError:
        print("\n⏰ 2-minute test completed!")
        engine.stop()
    except KeyboardInterrupt:
        print("\n🛑 Test stopped by user")
        engine.stop()


async def custom_test():
    """Run a custom test with user parameters"""
    
    print("🛠️ Custom Trading Test")
    print("=" * 30)
    
    try:
        # Get user inputs
        symbol = input("Symbol (default BTCUSDT): ").strip().upper() or "BTCUSDT"
        balance = float(input("Initial balance (default 500): ") or "500")
        position_size = float(input("Position size (default 50): ") or "50")
        
        # Create strategy with conservative parameters
        strategy = SimpleOrderBookStrategy(
            position_size=position_size,
            min_wall_size=150000,    # High threshold for quality signals
            min_pressure_diff=0.25,  # Strong pressure required
            cooldown_seconds=180,    # 3 minute cooldown
            min_hold_seconds=90      # Hold for at least 1.5 minutes
        )
        
        # Create engine
        engine = LiveTradingEngine(
            strategy=strategy,
            initial_balance=balance,
            stats_interval=25
        )
        
        print(f"\n🚀 Starting custom test with {symbol}")
        print("Press Ctrl+C to stop\n")
        
        await engine.start(symbol)
        
    except ValueError:
        print("❌ Invalid input. Please use numbers for balance and position size.")
    except KeyboardInterrupt:
        print("\n🛑 Custom test stopped")


def main():
    """Main test runner"""
    
    print("🤖 Trading System Test Runner")
    print("=" * 35)
    print("Choose a test:")
    print("1. Quick test (2 minutes, automated)")
    print("2. Custom test (configure parameters)")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        asyncio.run(quick_test())
    elif choice == "2":
        asyncio.run(custom_test())
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()